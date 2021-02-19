import torch
import numpy as np
import torch.nn as nn
import os 

class SurfaceNormals_new(nn.Module):
    
    def __init__(self):
        super(SurfaceNormals_new, self).__init__()
        
    def batch_arange(self, start, stop, step=1.):
        dtype = start.dtype
        device = start.device
        assert (stop >= start).all(), 'stop value should be greater or equal to start value'
        N = (stop - start) // step
        assert (N == N[0]).all(), 'all ranges have to be same length'
        N = N[0]
        steps = torch.empty_like(start)
        steps[:] = step
        return start[:, None] + steps[:, None] * torch.arange(N, dtype=dtype, device=device)
    
    def batch_meshgrid(self, h_range, w_range):
        N = h_range.shape[-1]
        M = w_range.shape[-1]
        h = h_range[..., None].expand(-1, -1, M)
        w = w_range[:, None, :].expand(-1, N, -1)
        return h, w
    
    def pc_to_normals(self, coords, order2=True):
        """Calculate surface normals using first order finite-differences.

        Parameters
        ----------
        coords : array_like
            Coordinates of the points (**, 3, h, w).

        Returns
        -------
        normals : torch.Tensor
            Surface normals (**, 3, h, w).
        """
        assert coords.dtype == torch.float64
        if order2:
            dxdu = self.gradient_for_normals(coords[:, 0], axis=2)
            dydu = self.gradient_for_normals(coords[:, 1], axis=2)
            dzdu = self.gradient_for_normals(coords[:, 2], axis=2)
            dxdv = self.gradient_for_normals(coords[:, 0], axis=1)
            dydv = self.gradient_for_normals(coords[:, 1], axis=1)
            dzdv = self.gradient_for_normals(coords[:, 2], axis=1)
        else:
            dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
            dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
            dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
            dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
            dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
            dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]
    
            dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode='replicate')
            dydu = torch.nn.functional.pad(dydu, (0, 1), mode='replicate')
            dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode='replicate')

            # pytorch cannot just do `dxdv = torch.nn.functional.pad(dxdv, (0, 0, 0, 1), mode='replicate')`, so
            dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
            dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
            dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

        n_x = dydv * dzdu - dydu * dzdv
        n_y = dzdv * dxdu - dzdu * dxdv
        n_z = dxdv * dydu - dxdu * dydv

        n = torch.stack([n_x, n_y, n_z], dim=-3)
        n = torch.nn.functional.normalize(n, dim=-3)
        return n
        
    def batch_pc(self, depth, depth_type, h, h_, w, w_, K, shift):
        
        dtype = depth.dtype
        assert dtype == torch.float64
        K = torch.as_tensor(K, dtype=dtype)
        h = torch.as_tensor(h, dtype=dtype)
        h_ = torch.as_tensor(h_, dtype=dtype)
        w = torch.as_tensor(w, dtype=dtype)
        w_ = torch.as_tensor(w_, dtype=dtype)
        
        v, u = self.batch_meshgrid(self.batch_arange(h, h_) + shift, self.batch_arange(w, w_ ) + shift)
        ones = torch.ones_like(v)
        points = torch.einsum('blk,bkij->blij', K.inverse(), torch.stack([u, v, ones],dim=1))
        if depth_type == 'orthogonal':
            points = points / points[:, 2:3]
#             print(points.to(depth).shape, depth.shape)
            points = points.to(depth) * depth
#         elif depth_type == 'perspective':
#             points = torch.nn.functional.normalize(points, dim=-3)
#             points = points.to(depth) * depth
#         elif depth_type == 'disparity':
#             points = points / points[2:3]
#             z = calibration['baseline'] * K[0, 0] / depth
#             points = points.to(depth) * z
        else:
            raise ValueError(f'Unknown type {depth_type}')
        return points
    
    def forward(self, depth, K, crop, depth_type='orthogonal', shift=.5):
        depth = depth.type(torch.float64)
        depth = (depth + 1.) / 2.
        h, h_, w, w_ = crop[:, 0], crop[:, 1], crop[:, 2], crop[:, 3]
        point = self.batch_pc(depth, depth_type, h, h_, w, w_, K, shift)
        return self.pc_to_normals(point).type(torch.float32)
#         dzdx = -self.gradient_for_normals(depth, axis=2)
#         dzdy = -self.gradient_for_normals(depth, axis=3)
#         norm = torch.cat((dzdx, dzdy, torch.ones_like(depth)), dim=1)
#         n = torch.norm(norm, p=2, dim=1, keepdim=True)
#         return torch.div(norm, torch.add(n, 1e-6))
    
    def gradient_for_normals(self, f, axis=None):
        N = f.ndim  # number of dimensions
        dx = 1.0
    
        # use central differences on interior and one-sided differences on the
        # endpoints. This preserves second order-accuracy over the full domain.
        # create slice objects --- initially all are [:, :, ..., :]
        slice1 = [slice(None)]*N
        slice2 = [slice(None)]*N
        slice3 = [slice(None)]*N
        slice4 = [slice(None)]*N
    
        otype = f.dtype
        if otype is torch.float32 or torch.float64:
            pass
        else:
            raise TypeError('Input shold be torch.float32')
    
        # result allocation
        out = torch.empty_like(f, dtype=otype)
    
        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
    
        out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * dx)
    
        # Numerical differentiation: 1st order edges
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        dx_0 = dx 
        # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        dx_n = dx 
        # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        return out
    
    
def get_imp_matrx(f_name):
        intrinsic_mtrx_path = '/mnt/neuro/un_depth/Scannet/'
        K = np.loadtxt(os.path.join(intrinsic_mtrx_path, f_name[:12], 'intrinsic', 'intrinsic_depth.txt'))[:3,:3]
        return K
    

def crop_indx(self, f_name):
        i, j = f_name.split('_')[3:]
        i, j = int(i), int(j)
        h_start = 64 * i + 5
        h_stop = h_start + 320
        w_start = 64 * j + 5
        w_stop = w_start + 320
        return h_start, h_stop, w_start, w_stop
    
    
    
    
    
class SurfaceNormals(nn.Module):
    
    def __init__(self):
        super(SurfaceNormals, self).__init__()
    
    def forward(self, depth):
        dzdx = -self.gradient_for_normals(depth, axis=2)
        dzdy = -self.gradient_for_normals(depth, axis=3)
        norm = torch.cat((dzdx, dzdy, torch.ones_like(depth)), dim=1)
        n = torch.norm(norm, p=2, dim=1, keepdim=True)
        return norm / (n + 1e-6)
    
    def gradient_for_normals(self, f, axis=None):
        N = f.ndim  # number of dimensions
        dx = 1.0
    
        # use central differences on interior and one-sided differences on the
        # endpoints. This preserves second order-accuracy over the full domain.
        # create slice objects --- initially all are [:, :, ..., :]
        slice1 = [slice(None)]*N
        slice2 = [slice(None)]*N
        slice3 = [slice(None)]*N
        slice4 = [slice(None)]*N
    
        otype = f.dtype
        if otype is torch.float32:
            pass
        else:
            raise TypeError('Input shold be torch.float32')
    
        # result allocation
        out = torch.empty_like(f, dtype=otype)
    
        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
    
        out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2. * dx)
    
        # Numerical differentiation: 1st order edges
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        dx_0 = dx 
        # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        dx_n = dx 
        # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
        out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        return out
