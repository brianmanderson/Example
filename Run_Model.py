__author__ = 'Brian M Anderson'
# Created on 1/18/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
from tensorflow.python.keras.models import *
import tensorflow.compat.v1 as tf
from Base_Deeplearning_Code.Keras_Utils.Keras_Utilities import balanced_cross_entropy, \
    weighted_categorical_crossentropy, categorical_crossentropy_masked, dice_coef_3D, np, EarlyStopping_BMA
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow.python.keras.backend as K
from Base_Deeplearning_Code.Callbacks.Visualizing_Model_Utils import TensorBoardImage
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Data_Generators.Return_Paths import Path_Return_Class
from tensorflow.python.keras.optimizers import Adam, SGD
from Base_Deeplearning_Code.Models.Keras_Models import my_UNet
from Base_Deeplearning_Code.Callbacks.BMA_Callbacks import ModelCheckpoint_new, Add_LR_To_Tensorboard
from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback import CyclicLR
from Return_Train_Validation_Generators import return_generators, get_layers_dict_atrous, return_dictionary,\
    return_dictionary_best, return_dictionary_cube, return_base_dict


def get_layers_dict(layers=1, filters=16, conv_blocks=1, num_atrous_blocks=4, max_blocks=2, max_filters=np.inf,
                    atrous_rate=2, max_atrous_rate=2, **kwargs):
    # activation = {'activation':PReLU,'kwargs':{'alpha_initializer':Constant(0.25),'shared_axes':[1,2,3]}}
    activation = 'elu'
    layers_dict = {}
    conv_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (1, 1, 1),'activation':activation}}
    strided_block = lambda x: {'convolution': {'channels': x, 'kernel': (3, 3, 3), 'strides': (2, 2, 2), 'activation':activation}}
    atrous_block = lambda x,y,z: {'atrous': {'channels': x, 'atrous_rate': y, 'activations': z}}
    for layer in range(conv_blocks,layers-1):
        encoding = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        atrous_block_dec = [atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]
        if layer == 0:
            encoding = [conv_block(filters)] + encoding
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)] = {'Encoding': encoding,
                                              'Pooling':{'Encoding':[strided_block(filters)],'Pool_Size':(2,2,2)},
                                              'Decoding': atrous_block_dec}
        num_atrous_blocks = min([(num_atrous_blocks) * 2,max_blocks])
    num_atrous_blocks = min([(num_atrous_blocks) * 2, max_blocks])
    layers_dict['Base'] = {'Encoding':[atrous_block(filters,atrous_rate,[activation for _ in range(atrous_rate)]) for _ in range(num_atrous_blocks)]}
    return layers_dict


def return_things(run_data):
    things = []
    for top_key in ['Architecture','Hyper_Parameters']:
        model_info = run_data[top_key]
        for key in model_info:
            if model_info[key] is not 0 and model_info[key] is not False and model_info[key] is not None:
                if model_info[key] is True:
                    things.append('{}'.format(key))
                else:
                    things.append('{}_{}'.format(model_info[key],key))
    return things


def run_model(min_lr=1e-4, max_lr=1e-2, layers_dict=None, epochs=1000,validation_generator=None,step_size=None,paths_class=None,
              step_size_factor=5, train_generator=None, batch_norm=False,mask_pred=False,pre_cycle=0,write_images=True,load_path=None,
              morfeus_drive='',base_path='', save_a_model=True,weighted=False, balance_beta=1.0, epoch_i = 0, sgd=False,
              model_params=None,skip_cyclic_lr=False, scale_mode='linear_cycle',**kwargs):
    if step_size is None:
        step_size = len(train_generator)
    with tf.device('/gpu:0'):
        gpu_options = tf.GPUOptions(allow_growth=True) # maybe should just allocate whole gpu..
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        K.set_session(sess)
        if not os.path.exists(morfeus_drive):
            print('Morf wrong')
            return None
        if not os.path.exists(base_path):
            print('base wrong')
            return None


        model_path_out = paths_class.model_path_out
        tensorboard_output = paths_class.tensorboard_path_out
        if not sgd:
            optimizer = Adam(lr=min_lr)
        else:
            optimizer = SGD(lr=min_lr)
        print('Learning rate is {}'.format(min_lr))
        wait = 1
        period = 10
        # if save_a_model:
        #     period = 5
        monitor = 'val_loss' #dice_coef_3D
        mode = 'min'
        checkpoint = ModelCheckpoint_new(model_path_out, monitor=monitor, verbose=1, save_best_only=False,save_best_and_all=True,
                                         save_weights_only=False, period=period, mode=mode)
        tensorboard = TensorBoardImage(log_dir=tensorboard_output, write_graph=True, write_grads=False,num_images=1,
                                       update_freq='epoch',  data_generator=validation_generator, image_frequency=period,
                                       write_images=write_images)
        lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=step_size, step_size_factor=step_size_factor, mode='triangular2',
                         pre_cycle=pre_cycle, base_reduce_factor=2, scale_mode=scale_mode,
                         step_size_factor_scale=lambda x: x + 0)
        early_stopping = EarlyStopping_BMA(monitor=monitor,min_delta=0,patience=15,verbose=1,mode=mode,
                                           max_delta=1.0,baseline=2.2,restore_best_weights=True,wait=wait)
        # early_stopping = EarlyStopping(monitor=monitor, patience=15, verbose=1, mode=mode)
        callbacks = [Add_LR_To_Tensorboard(), tensorboard] #early_stopping, lrate,
        if not skip_cyclic_lr:
            callbacks += [lrate]
        if save_a_model:
            callbacks += [checkpoint]
        loss = 'categorical_crossentropy'
        if weighted:
            loss = weighted_categorical_crossentropy(np.asarray([1,500])) #categorical_crossentropy
            print('weighted loss')
        if balance_beta != 1.0:
            loss = balanced_cross_entropy(balance_beta)
        if load_path is None:
            model = my_UNet(kernel=(3, 3, 3), layers_dict=layers_dict, pool_size=(2, 2, 2),batch_norm=batch_norm,
                            pool_type='Max',out_classes=2, mask_output=mask_pred,**model_params)
            Model_val = model.created_model
        else:
            print('\n\n\n\nLoading model at {}\n\n\n\n'.format(load_path))
            Model_val = load_model(load_path, custom_objects={'dice_coef_3D': dice_coef_3D})
        if os.listdir(tensorboard_output):
            print('already done')
            return None
        print('\n\n\n\nRunning {}\n\n\n\n'.format(tensorboard_output))
        Model_val.compile(optimizer, loss=loss, metrics=['accuracy', dice_coef_3D])
        Model_val.fit_generator(generator=train_generator, workers=10, use_multiprocessing=False, max_queue_size=50,
                                shuffle=True, epochs=epochs, callbacks=callbacks, initial_epoch=epoch_i,
                                validation_data=validation_generator,steps_per_epoch=step_size,
                                validation_freq=period)


def train_model(epochs=None,run_best=False, save_a_model=False, path_extension='Single_Images3D_1mm',
                cube_size=(8, 20, 120, 120),model_name = '3D_Fully_Atrous', step_size_factor=10, sgd=False):
    mask_image = False
    mask_loss = False
    mask_pred = True
    batch_norm = False
    write_images = True
    norm_to_liver = True
    smoothing = 0.0
    weighted = False

    num_patients = 1
    base_path, morfeus_drive, train_generator, validation_generator = return_generators(liver_norm=norm_to_liver,
                                                                                        cube_size=cube_size,
                                                                                        path_extension=path_extension,
                                                                                        num_patients=num_patients)
    print(base_path)
    x,y = train_generator.__getitem__(0)
    epoch_i = 0
    num_cycles = 10
    step_size = len(train_generator)
    base_dict = return_base_dict(step_size_factor=step_size_factor,
                                 save_a_model=save_a_model, sgd_opt=sgd)
    if epochs is None:
        epochs = step_size_factor
        for _ in range(1,num_cycles):
            epochs += (step_size_factor * 2)
        epochs += 2
        epochs = min([1000,epochs])
        epochs = max([300, epochs])
    model_params = {'activation':'elu', 'concat_not_add':False}

    if smoothing > 0:
        model_name += '{}_smoothing'.format(smoothing)
    for iteration in range(3):
        if run_best:
            overall_dictionary = return_dictionary_best(base_dict, sgd=sgd)
        else:
            overall_dictionary = return_dictionary_cube(base_dict)
        for run_data in overall_dictionary:
            run_data['Architecture']['model_name'] = model_name
            run_data['Architecture']['batch_norm'] = batch_norm
            run_data['Architecture']['mask_image'] = mask_image
            run_data['Architecture']['mask_pred'] = mask_pred
            run_data['Architecture']['mask_loss'] = mask_loss
            things = return_things(run_data)
            things += ['{}_Iteration'.format(iteration)]
            layers_dict = get_layers_dict_atrous(**run_data['Architecture'])
            # while True:
            #     for i in range(5):
            #         x,y = validation_generator_3D.__getitem__(i)
            paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive, save_model=save_model)
            paths_class.define_model_things(model_name, things)
            tensorboard_output = paths_class.tensorboard_path_out
            print(tensorboard_output)
            if os.listdir(tensorboard_output):
                print('already done')
                continue
            try:
                run_model(layers_dict=layers_dict, train_generator=train_generator,
                          step_size=step_size,epoch_i=epoch_i,
                          validation_generator=validation_generator,save_a_model=save_a_model,
                          model_params=model_params, paths_class=paths_class,morfeus_drive=morfeus_drive,
                          base_path=base_path, epochs=epochs, weighted=weighted,sgd=sgd,
                          write_images=write_images,**run_data['Architecture'],**run_data['Hyper_Parameters'])
                K.clear_session()
            except:
                print('failed here')
                K.clear_session()


if __name__ == '__main__':
    pass
