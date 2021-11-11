__author__ = 'Brian M Anderson'
# Created on 3/23/2020

import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

cube_size = (16, 16, 120, 120)
batch_size = 8
add = '_32'
path_desc='TF_LR_Dense_1mm_multiconv'
model_name = 'TF2_3D_Fully_Atrous_Variable_Cube_Training_1mm'
cache_add = '_1mm'

find_dense_lr_dense = True
if find_dense_lr_dense:
    from Optimization.Find_Best_LR_TF2_Dense import find_best_lr
    find_best_lr(batch_size=batch_size, path_desc=path_desc, add=add, cache_add=cache_add)

find_lr = False
if find_lr:
    from Optimization.Find_Best_LR_TF2 import find_best_lr
    find_best_lr(optimizer='Adam', batch_size=batch_size, path_desc=path_desc, add=add)
'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots
    from Return_Train_Validation_Generators_TF2 import return_paths
    _, morfeus_drive = return_paths()
    path = os.path.join(morfeus_drive,path_desc, 'Dense')
    make_plots(path)

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = False
if run_200:
    from Run_Model_TF2 import train_model
    run_best = True
    train_model(epochs=201, model_name=model_name, run_best=run_best, debug=False, add=add, dense=True,
                cache_add=cache_add, change_background=False)

make_opt_excel = False
if make_opt_excel:
    '''
    Need to run the model for ~ 200 epochs, then run Plot_Optimization_results
    '''
    from Optimization.Plot_Optimization_results_TF2 import main
    from Return_Train_Validation_Generators_TF2 import return_paths
    base_path, morfeus_drive = return_paths()
    main()
