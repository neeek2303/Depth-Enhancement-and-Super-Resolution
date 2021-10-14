import time
from options.train_options import TrainOptions
from data import create_dataset
from util.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
from models.main_model import MainModel
from models.main_sr_model import MainSRModel
from models.I2D_model import I2DModel
from models.translation_model import TranslationModel
import torch
from collections import OrderedDict 
from plots import plot_main, plot_I2D, plot_translation

def sum_of_dicts(dict1, dict2, l):
    
    output = OrderedDict([(key, dict1[key]+dict2[key]/l) for key in dict1.keys()])
    return output


if __name__ == '__main__':
    opt = TrainOptions().parse()   
    
    if opt.use_wandb:
        import wandb
        wandb.init(project="translation_compare")
        wandb.config.update(opt)
    
    if opt.model_type == "I2D":
        plot_function = plot_I2D
        model = I2DModel(opt)
        from data.my_I2D_dataset import MyUnalignedDataset
    elif opt.model_type == "main":
        plot_function = plot_main
        model = MainModel(opt)
        from data.my_main_dataset import MyUnalignedDataset
        if opt.SR:
            model = MainSRModel(opt)
            from data.my_naive_sr_dataset import MyUnalignedDataset
    elif opt.model_type == "translation":
        model = TranslationModel(opt)
        from data.translation_dataset import MyUnalignedDataset
        plot_function = plot_translation
        
        
        
    dataset = create_dataset(opt, MyUnalignedDataset) 
    test_dataset = create_dataset(opt, MyUnalignedDataset, stage='test')
    print(len(dataset), len(test_dataset))
    
    dataset_size = len(dataset)    
    print('The number of training images = %d' % dataset_size)
 
    model.setup(opt)               

    total_iters = opt.start_iter                
    test_iter=0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + opt.epoch_count):    
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0  
        
        if opt.do_train:
            model._train()
            stage = 'train'
            for i, data in enumerate(dataset): 
                iter_start_time = time.time()  
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         
                model.optimize_parameters(total_iters, opt.update_ratio)  

                if (total_iters-opt.start_iter) % opt.display_freq == 0 and  opt.use_wandb:   
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    image_dict = model.get_current_visuals()
                    depth = opt.input_nc == 1
                    plot_function(wandb, image_dict, total_iters, depth=depth, stage=stage)


                if (total_iters-opt.start_iter) % opt.print_freq == 0:    
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    if  opt.use_wandb:
                        wandb.log(losses, step = total_iters)
                    else:
                        print('stage: ', stage)
                        print(losses)
                        print()


                if (total_iters-opt.start_iter) % opt.save_latest_freq*opt.batch_size == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                    print('metrics')

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate() 
        
        if opt.do_test:
            model.eval()
            stage = 'test'
            if opt.model_type == "I2D":
                mean_losses = OrderedDict([('task_syn', 0.0), ('task_real', 0.0)])
            elif opt.model_type == "main":    
                mean_losses = OrderedDict([('task_syn', 0.0), ('holes_syn', 0.0), ('task_real_by_depth', 0.0), ('holes_real', 0.0), ('syn_norms', 0.0) ])
                
            with torch.no_grad():   
                l = len(test_dataset)
                for i, data in enumerate(test_dataset):  # inner loop within one epoch
                    test_iter += opt.batch_size
                    model.set_input(data)
                    model.calculate(stage = stage)
                    
                    if not opt.SR:
                        losses = model.get_current_losses()
                        mean_losses = sum_of_dicts(mean_losses, losses,  l/opt.batch_size_val)

                if opt.use_wandb:
                    wandb.log({stage:mean_losses}, step = total_iters)
                print('stage: ', stage)
                print(mean_losses)
                print('=====================================================================================')
 
      
        
