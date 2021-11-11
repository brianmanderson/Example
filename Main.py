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

path_extension = 'Single_Images3D_None'
cube_size = (8, 32, 100, 100)
path_desc='3.25_Learning_Rates_Cube_Training'
model_name = '3D_Fully_Atrous_Variable_Cube_Training_8_32_100_100'
'''
Plot the LR, get the min and max from the images
'''
plot_lr = False
if plot_lr:
    from Optimization.Plot_Best_LR import make_plots
    from Return_Train_Validation_Generators import return_generators
    _, morfeus_drive, _, _ = return_generators(path_extension=path_extension)
    path = os.path.join(morfeus_drive,path_desc)
    make_plots(path)
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''

'''
Now go to Evaluate_Model folder
'''

'''
Now, we need to run the model for a number of epochs ~200, so we can get a nice curve to make final model
decision based on
'''
run_200 = True
if run_200:
    from Run_Model import train_model
    sgd=False
    for step_size_factor in [20]:
        for add in [10]:
            train_model(epochs=1005, step_size_factor=step_size_factor,
                        save_a_model=True, run_best=True, path_extension=path_extension,
                        cube_size=cube_size, model_name=model_name, sgd=sgd)
