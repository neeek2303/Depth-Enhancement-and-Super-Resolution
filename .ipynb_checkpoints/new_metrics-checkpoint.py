import os
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import imageio
from scipy import signal
import tqdm
from glob import glob
import multiprocessing
import functools
import torch
import albumentations as A
holes_threshold = 50

filter_basename = lambda x: os.path.splitext(os.path.basename(x))[0]

def coords_to_normals(coords):
    coords = torch.as_tensor(coords)
    if coords.ndim < 4:
        coords = coords[None]

    dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
    dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
    dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
    dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
    dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
    dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

    dxdu = torch.nn.functional.pad(dxdu, (0, 1),       mode='replicate')
    dydu = torch.nn.functional.pad(dydu, (0, 1),       mode='replicate')
    dzdu = torch.nn.functional.pad(dzdu, (0, 1),       mode='replicate')

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

def depth_to_absolute_coordinates(depth, depth_type, K, shift):

    depth = torch.as_tensor(depth)
    dtype = depth.dtype
    h, w = depth.shape[-2:]
    K = torch.as_tensor(K, dtype=dtype)

    v, u = torch.meshgrid(torch.arange(h, dtype=dtype) + shift, torch.arange(w, dtype=dtype) + shift)
    if depth.ndim < 3:  # ensure depth has channel dimension
        depth = depth[None]
    ones = torch.ones_like(v)
    points = torch.einsum('lk,kij->lij', K.inverse(), torch.stack([u, v, ones]))
    if depth_type == 'perspective':
        points = torch.nn.functional.normalize(points, dim=-3)
        points = points.to(depth) * depth
    elif depth_type == 'orthogonal':
        points = points / points[2:3]
        points = points.to(depth) * depth
    else:
        raise ValueError(f'Unknown type {depth_type}')
    return points

def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def _ssim(img1, img2, L=1):
    """
        L = 1 for depth maps in [0, 1] range
    
        Return the Structural Similarity Map corresponding to input images img1 
        and img2 (images are assumed to be uint8)
        This function attempts to mimic precisely the functionality of ssim.m a 
        MATLAB provided by the author's of SSIM
        https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    return np.mean(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)))
    

def _mse(pred, target):
    return  np.mean(np.power(target - pred, 2))

def _rmse(pred, target):
    return np.sqrt(_mse(pred, target))

def _psnr(pred, target, max_value=1):
    mse = np.mean(np.power(target - pred, 2))
    if mse == 0:
        raise NotImplementedError('Same img')
    else:
        return 20. * np.log10(max_value) - 10 * np.log10(mse)

def _mae(pred, target):
    return np.mean(np.abs(target - pred))

def calc_mae_h(pred, target, hole_map, target_hole_map, K, max_depth):
    pred_no_target_hole_map = ~target_hole_map * hole_map # map of holes in pred, but where no holes in target
    if np.any(pred_no_target_hole_map):
        return np.mean(np.abs(pred[pred_no_target_hole_map] - target[pred_no_target_hole_map]))
    else:
        return np.nan

def calc_mae_d(pred, target, hole_map, target_hole_map, K, max_depth):
    union_hole_map = hole_map + target_hole_map
    if not np.all(union_hole_map):
        return np.mean(np.abs(pred[~union_hole_map]- target[~union_hole_map]))
    else:
        return np.nan

def calc_rmse_h(pred, target, hole_map, target_hole_map, K, max_depth):
    pred_no_target_hole_map = ~target_hole_map * hole_map # map of holes in pred, but where no holes in target
    if np.any(pred_no_target_hole_map):
        diff2 = np.square(pred[pred_no_target_hole_map] - target[pred_no_target_hole_map])
        return np.sqrt(np.mean(diff2))
    else:
        return np.nan
    
def calc_rmse_d(pred, target, hole_map, target_hole_map, K, max_depth):
    union_hole_map = hole_map + target_hole_map
    if not np.all(union_hole_map):
        diff2 = np.square(pred[~union_hole_map]- target[~union_hole_map])
        return np.sqrt(np.mean(diff2))
    else:
        return np.nan

def calc_msev(pred, target, hole_map, target_hole_map, K, max_depth, depth_type='orthogonal', shift=0.5):
    target_pc = depth_to_absolute_coordinates(target, depth_type, K, shift)
    pred_pc = depth_to_absolute_coordinates(pred, depth_type, K, shift)
    
    target_n = coords_to_normals(target_pc).squeeze(0).numpy()
    pred_n = coords_to_normals(pred_pc).squeeze(0).numpy()
    
    target_normal_hole_map = target_hole_map.copy() # we need to make the map 1 pixel wider
    target_normal_hole_map[:, 1:] += target_hole_map[:, :-1]
    target_normal_hole_map[:, :-1] += target_hole_map[:, 1:]
    target_normal_hole_map[1:, :] += target_hole_map[:-1, :]
    target_normal_hole_map[:-1, :] += target_hole_map[1:, :]
    target_normal_hole_map = np.broadcast_to(target_normal_hole_map, pred_n.shape)
    return _mse(pred_n[~target_normal_hole_map], target_n[~target_normal_hole_map])

def calc_rmse(pred, target, hole_map, target_hole_map, K, max_depth):
    return _rmse(pred[~target_hole_map], target[~target_hole_map])

def calc_psnr(pred, target, hole_map, target_hole_map, K, max_depth):
    return _psnr(pred[~target_hole_map]/max_depth, target[~target_hole_map]/max_depth) 

def calc_mae(pred, target, hole_map, target_hole_map, K, max_depth):
    return _mae(pred[~target_hole_map], target[~target_hole_map])

def calc_ssim(pred, target, hole_map, target_hole_map, K, max_depth):
    return _ssim(~target_hole_map * pred/max_depth, ~target_hole_map * target/max_depth) 

metric_by_name = {
    "mae": calc_mae,
    "rmse": calc_rmse,
    "psnr": calc_psnr,
    "ssim": calc_ssim,
    "rmse_h": calc_rmse_h,
    "rmse_d": calc_rmse_d,
    "mae_h": calc_mae_h,
    "mae_d": calc_mae_d,
    "mse_v": calc_msev,
}

def calc_metrics(pred, target, hole_map, target_hole_map, K, max_depth, metric_names):
    out = {}
    
    for metric_name in metric_names:
        metric_func = metric_by_name[metric_name]
        out[metric_name] = metric_func(pred, target, hole_map, target_hole_map, K, max_depth)

    return out

def apply_transformer(transformations,  depth):
    res = A.Compose(transformations, p=1)(image=depth)
    return res

def calc_metrics_for_path(path_args, metric_names, max_depth):
    input_path, pred_path, target_path, intrisic_path = path_args
    input_orig = imageio.imread(input_path).astype(np.float64)
    pred = imageio.imread(pred_path).astype(np.float64).clip(0, max_depth) 
    target = imageio.imread(target_path).astype(np.float64).clip(0, max_depth)
    h_pred, w_pred = pred.shape
    h_target, w_target = target.shape
            
    h_pred, w_pred = pred.shape
    h_target, w_target = target.shape    

    if 2*h_pred == h_target: # if our target is 2x bigger than prediction
        target = target[0::2, 0::2]
    hole_map = input_orig < holes_threshold
    target_hole_map = target < holes_threshold
    K = np.loadtxt(intrisic_path)[:3,:3] if intrisic_path is not None else None
#     K[0][0]=K[0][0]*2
#     K[1][1]=K[1][1]*2
#     K[1][2]=K[1][2]*2
#     K[0][2]=K[0][2]*2
    scale_K = np.array([[2., 1., 2.],[1., 2., 2.],[1., 1., 1.]])
    return calc_metrics(pred, target, hole_map, target_hole_map, K, max_depth, metric_names)

def calculate_given_paths(input_names, pred_names, target_names, metric_names, max_depth, n_cpus):

    print(len(input_names), len(pred_names), len(target_names))
    #check that filenames are the same
    
    intrinsic_names = list(map(lambda x: os.path.join('/root/datasets/un_depth/Scannet', x[:12], 'intrinsic', 'intrinsic_depth.txt'),
                               (filter_basename(input_name) for input_name in input_names)))
    _calc_metrics_for_path = functools.partial(calc_metrics_for_path, metric_names=metric_names, max_depth=max_depth)
    paths = zip(input_names, pred_names, target_names, intrinsic_names)
    with multiprocessing.Pool(n_cpus) as p:
        res = list(p.imap(func=_calc_metrics_for_path, iterable=paths))
    out = {}
    for metric_name in metric_names:
        out[metric_name] = np.asarray([x[metric_name] for x in res])
        out[metric_name] = np.mean(out[metric_name][~np.isnan(out[metric_name])])

    return out


if __name__ == '__main__':
    print('start')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', type=str, default = '/root/datasets/un_depth/Scannet_ssim/testA/full_size/depth', help='Path to the input images')
    parser.add_argument('--pred_path', type=str, default = '/root/code_for_article/depth_SR/test_pred', help='Path to the generated images')
    parser.add_argument('--target_path', type=str, default = '/root/datasets/un_depth/Scannet_ssim/testB/full_size/depth', help='Path to the target images')
    parser.add_argument('--max_depth', type=int, default=5100, help='Maximum depth value')
    parser.add_argument('--n_cpus', type=int, default=10, help='Number of cpu cores to use')
    args = parser.parse_args()

    
    input_names = sorted(glob(os.path.join(args.input_path,'*.png')))
    pred_names = sorted(glob(os.path.join(args.pred_path,'*.png')))
    target_names = sorted(glob(os.path.join(args.target_path,'*.png')))
    list_of_metrics = ["rmse", "mae", "rmse_h", "rmse_d", "psnr", "ssim", "mae_h", "mae_d", "mse_v"]
    out = calculate_given_paths(input_names, pred_names, target_names, list_of_metrics, args.max_depth, 10)
    print(out)
    
#     for i in range(len(input_names)):
#         print(input_names[i], pred_names[i], target_names[i])
#         out = calculate_given_paths([input_names[i]], [pred_names[i]], [target_names[i]], list_of_metrics, args.max_depth, 30)
#         print(out)