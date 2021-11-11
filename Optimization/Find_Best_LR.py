import sys
sys.path.append('..')
from Base_Deeplearning_Code.Data_Generators.Image_Processors import *
import tensorflow.python.keras.backend as K
import tensorflow.compat.v1 as tf
from Base_Deeplearning_Code.Models.Keras_Models import my_UNet
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import dice_coef_3D
from tensorflow.python.keras.optimizers import SGD
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder, Adam
from Return_Train_Validation_Generators import return_generators, return_base_dict, get_layers_dict_atrous, OrderedDict
from tensorflow.python.keras.callbacks_v1 import TensorBoard


def run_model(layers_dict=None, out_path='',train_generator=None):
    x,y = train_generator.__getitem__(0)
    print(len(train_generator))
    with tf.device('/gpu:0'):
        gpu_options = tf.GPUOptions()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        loss = 'categorical_crossentropy'
        model = my_UNet(kernel=(3, 3, 3), layers_dict=layers_dict, pool_type='Max', out_classes=2, mask_output=True
                        ).created_model
        # k = TensorBoard(log_dir=out_path)
        # k.set_model(model)
        epochs = 10
        print('\n\n{}\n\n'.format(out_path))
        LearningRateFinder(epochs=epochs, model=model, metrics=['accuracy', dice_coef_3D],out_path=out_path,loss=loss,
                           train_generator=train_generator, lower_lr=1e-7, high_lr=1e-2, optimizer=SGD)


def return_things(run_data):
    things = []
    for top_key in ['Architecture']:
        model_info = run_data[top_key]
        for key in model_info:
            if model_info[key] is not False:
                if model_info[key] is True:
                    things.append('{}'.format(key))
                else:
                    things.append('{}_{}'.format(model_info[key],key))
    return things


def find_best_lr(path_extension='Single_Images3D_1mm', cube_size = (30,300,300), path_desc='3.25_Learning_Rates_New_Training'):
    base_path, morfeus_drive, train_generator, validation_generator = return_generators(path_extension=path_extension,
                                                                                        cube_size=cube_size)
    x,y = train_generator.__getitem__(0)
    base_dict = return_base_dict(sgd_opt=False)
    min_lr = 1e-7
    max_lr = 1e-2
    for iteration in [0, 1, 2]:
        for layer in [1,2,3,4,5]: #
            for filters in [8, 16]: #, 16
                for max_filters in [16, 32]: #, 32
                    run_data = base_dict(min_lr=min_lr, max_lr=max_lr, filters=filters, max_filters=max_filters,
                                         layers=layer)
                    layers_dict = get_layers_dict_atrous(**run_data['Architecture'])
                    things = return_things(run_data)
                    things.append('{}_Iteration'.format(iteration))
                    out_path = os.path.join(morfeus_drive,path_desc,'Fully_Atrous')
                    for thing in things:
                        out_path = os.path.join(out_path,thing)
                    if os.path.exists(out_path):
                        continue
                    print(out_path)
                    os.makedirs(out_path)
                    try:
                        run_model(layers_dict=layers_dict, out_path=out_path,
                                  train_generator=train_generator)
                        K.clear_session()
                    except:
                        K.clear_session()


if __name__ == '__main__':
    pass


