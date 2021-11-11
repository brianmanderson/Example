__author__ = 'Brian M Anderson'

# Created on 8/31/2020

import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
cube_size = (16, 120, 120)
from Return_Train_Validation_Generators_TF2 import return_paths

base_path, morfeus_drive = return_paths()
kernel = (3, 3)
batch_size = 12
squeeze_kernel = (1, 1)

add = '_16'
path_desc = 'TF_LR_2D_DenseNetMultiBatch'
excel_file_name = 'parameters_list_by_trial_id_DenseNetMultibatch.xlsx'
model_name = 'DenseNetNewMultiBatch'
cache_add = ''
path_lead = 'Records_1mm'
find_dense_lr_densenet121_pretrained = True
if find_dense_lr_densenet121_pretrained:
    from Optimization.Find_Best_LR_TF2_Dense import find_best_lr_DenseNet

    find_best_lr_DenseNet(batch_size=batch_size, path_desc=path_desc, add=add, cache_add=cache_add, path_lead=path_lead,
                          all_trainable=False, weights_path=None, layers_dict=None, model_name=model_name)

'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots

    path = os.path.join(morfeus_drive, path_desc, model_name)
    make_plots(path)

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200_pretrained = False
if run_200_pretrained:
    from Run_Model_TF2 import train_DenseNet

    run_best = False
    all_trainable = False
    train_DenseNet(epochs=101, model_name=model_name, run_best=run_best, add=add, cache_add=cache_add,
                   batch_size=batch_size,
                   change_background=False, path_lead=path_lead, validation_name='_64', all_trainable=all_trainable,
                   weights_path=None, layers_dict=None, excel_file_name=excel_file_name)

'''
Turn on the weights, and find a good learning rate
'''
all_trainable = True
weights_path = os.path.join(base_path, 'Keras', model_name, 'Models', 'Trial_ID_7', 'cp-0016.h5')

find_dense_lr_densenet121_retrained = False
if find_dense_lr_densenet121_retrained:
    from Optimization.Find_Best_LR_TF2_Dense import find_best_lr_DenseNet

    find_best_lr_DenseNet(batch_size=batch_size, path_desc=path_desc, add=add, cache_add=cache_add, path_lead=path_lead,
                          all_trainable=all_trainable, weights_path=weights_path, layers_dict=None,
                          model_name=model_name)

'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots

    path = os.path.join(morfeus_drive, path_desc, model_name)
    make_plots(path)

'''
Run with all weights turned on
'''
run_200_retrained = False
if run_200_retrained:
    from Run_Model_TF2 import train_DenseNet

    run_best = False
    train_DenseNet(epochs=101, model_name=model_name, run_best=run_best, add=add, cache_add=cache_add,
                   batch_size=batch_size,
                   change_background=False, path_lead=path_lead, validation_name='_64', all_trainable=all_trainable,
                   weights_path=weights_path, layers_dict=None, excel_file_name=excel_file_name)

excel_file_name = 'parameters_list_by_trial_id_DenseNetMultibatch3D.xlsx'
'''
Now slap on 3D and turn off trainable on 2D
'''
all_trainable = False
weights_path = os.path.join(base_path, 'Keras', model_name, 'Models', 'Trial_ID_13', 'cp-0101.h5')

find_dense_lr_densenet121_3D_pretrained = False
if find_dense_lr_densenet121_3D_pretrained:
    from Optimization.Find_Best_LR_TF2_Dense import find_best_lr_DenseNet3D

    find_best_lr_DenseNet3D(batch_size=batch_size, path_desc=path_desc, add=add, cache_add=cache_add,
                            path_lead=path_lead, all_trainable=all_trainable, weights_path=weights_path,
                            model_name=model_name)

'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots

    path = os.path.join(morfeus_drive, path_desc, model_name)
    make_plots(path)

'''
Now run it with the 3D model on and training
'''
run_200_retrained = False
if run_200_retrained:
    from Run_Model_TF2 import train_DenseNet3D

    run_best = False
    train_DenseNet3D(epochs=31, model_name=model_name, run_best=run_best, add=add, cache_add=cache_add,
                     batch_size=batch_size,
                     change_background=False, path_lead=path_lead, validation_name='_64', all_trainable=all_trainable,
                     weights_path=weights_path, excel_file_name=excel_file_name)

make_opt_excel = False
if make_opt_excel:
    '''
    Need to run the model for ~ 200 epochs, then run Plot_Optimization_results
    '''
    from Optimization.Plot_Optimization_results_TF2 import main
    from Return_Train_Validation_Generators_TF2 import return_paths

    base_path, morfeus_drive = return_paths()
    main()
