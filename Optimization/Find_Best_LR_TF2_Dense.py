__author__ = 'Brian M Anderson'

# Created on 4/26/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
import tensorflow as tf
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from tensorflow.keras.callbacks import TensorBoard
from Return_Train_Validation_Generators_TF2 import return_generators, return_paths, return_model, \
    get_layers_dict_dense_new, get_layers_dict_dense_HNet


def find_best_lr(batch_size=16, path_desc='', add='', cache_add='_1mm', kernel=(3, 3, 3), squeeze_kernel=(1, 1, 1),
                 image_size=(None, None, None, 1), pool=(2, 2, 2)):
    min_lr = 1e-7
    max_lr = 1
    for iteration in [0, 1, 2]:
        for growth_rate in [0]:
            for layer in [4]:
                for max_conv_blocks in [3]:
                    for filters in [96]:
                        for num_conv_blocks in [3]:
                            for conv_lambda in [1]:
                                base_path, morfeus_drive = return_paths()
                                run_data = {'layers': layer, 'max_conv_blocks': max_conv_blocks, 'filters': filters,
                                            'num_conv_blocks': num_conv_blocks, 'conv_lambda': conv_lambda,
                                            'growth_rate': growth_rate, 'kernel': kernel,
                                            'squeeze_kernel': squeeze_kernel, 'pool': pool}
                                layers_dict = get_layers_dict_dense_new(**run_data)
                                things = ['new', 'layers{}'.format(layer), 'max_conv_blocks_{}'.format(max_conv_blocks),
                                          'filters_{}'.format(filters), 'num_conv_blocks_{}'.format(num_conv_blocks),
                                          'conv_lambda_{}'.format(conv_lambda), 'growth_rate_{}'.format(growth_rate),
                                          'last_skip',
                                          '{}_Iteration'.format(iteration)]
                                out_path = os.path.join(morfeus_drive, path_desc, 'Dense')
                                for thing in things:
                                    out_path = os.path.join(out_path, thing)
                                if os.path.exists(out_path):
                                    print('already done')
                                    continue
                                base_path, morfeus_drive, train_generator, validation_generator = return_generators(
                                    batch_size=batch_size, add=add, threshold_val=10, change_background=False,
                                    cache_add=cache_add)
                                is_2D = False
                                if image_size != (None, None, None, 1):
                                    is_2D = True
                                model = return_model(layers_dict, is_2D=is_2D)
                                k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
                                k.set_model(model)
                                k.on_train_begin()
                                lr_opt = tf.keras.optimizers.Adam
                                print(out_path)
                                LearningRateFinder(epochs=10, model=model, metrics=['sparse_categorical_accuracy'],
                                                   out_path=out_path, optimizer=lr_opt,
                                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                                       from_logits=False),
                                                   steps_per_epoch=len(train_generator),
                                                   train_generator=train_generator.data_set, lower_lr=min_lr,
                                                   high_lr=max_lr)
                                tf.keras.backend.clear_session()
                                return None  # repeat!


def find_best_lr_DenseNet(batch_size=0, path_desc='', add='_16', cache_add='_1mm', path_lead='Records',
                          all_trainable=False, weights_path=None, layers_dict=None, model_name='DenseNet121'):
    min_lr = 1e-7
    max_lr = 1
    base_path, morfeus_drive = return_paths()
    for iteration in [0, 1, 2]:
        things = ['all_trainable_{}'.format(all_trainable)]
        things += ['3D_Model_{}'.format(layers_dict is not None)]
        things += ['{}_Iteration'.format(iteration)]
        out_path = os.path.join(morfeus_drive, path_desc, model_name)
        for thing in things:
            out_path = os.path.join(out_path, thing)
        if os.path.exists(out_path):
            print('already done')
            continue
        os.makedirs(out_path)
        base_path, morfeus_drive, train_generator, validation_generator = return_generators(
            batch_size=batch_size, add=add, threshold_val=10, change_background=False,
            cache_add=cache_add, path_lead=path_lead, validation_name='_64')
        model = return_model(layers_dict, weights_path=weights_path, densenet=True, all_trainable=all_trainable)
        k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
        k.set_model(model)
        k.on_train_begin()
        lr_opt = tf.keras.optimizers.Adam
        print(out_path)
        LearningRateFinder(epochs=10, model=model, metrics=['sparse_categorical_accuracy'],
                           out_path=out_path, optimizer=lr_opt,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           steps_per_epoch=len(train_generator),
                           train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
        tf.keras.backend.clear_session()
        return None  # repeat!


def find_best_lr_DenseNet3D(batch_size=0, path_desc='', add='_16', cache_add='_1mm', path_lead='Records',
                            all_trainable=False, weights_path=None, model_name=''):
    min_lr = 1e-7
    max_lr = 1
    base_path, morfeus_drive = return_paths()
    for iteration in [0]:
        for layers in [2, 3]:
            for num_conv_blocks in [2, 4]:
                for conv_lambda in [2, 4]:
                    if all_trainable:
                        layers = 2
                        num_conv_blocks = 2
                        conv_lambda = 2
                    layers_dict = get_layers_dict_dense_HNet(layers=layers, filters=32, num_conv_blocks=num_conv_blocks,
                                                             conv_lambda=conv_lambda,
                                                             max_conv_blocks=12)
                    things = ['all_trainable_{}'.format(all_trainable), '3D_Model_{}'.format(layers_dict is not None)]
                    things += ['layers_{}'.format(layers), 'conv_blocks_{}'.format(num_conv_blocks),
                               'lambda_{}'.format(conv_lambda)]
                    things += ['{}_Iteration'.format(iteration)]
                    out_path = os.path.join(morfeus_drive, path_desc, model_name)
                    for thing in things:
                        out_path = os.path.join(out_path, thing)
                    if os.path.exists(out_path):
                        print('already done')
                        continue
                    _, _, train_generator, validation_generator = return_generators(
                        batch_size=batch_size, add=add, threshold_val=10, change_background=False,
                        cache_add=cache_add, path_lead=path_lead, validation_name='_64')
                    os.makedirs(out_path)
                    model = return_model(layers_dict, weights_path=weights_path, densenet=True,
                                         all_trainable=all_trainable)
                    k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
                    k.set_model(model)
                    k.on_train_begin()
                    lr_opt = tf.keras.optimizers.Adam
                    print(out_path)
                    LearningRateFinder(epochs=10, model=model, metrics=['sparse_categorical_accuracy'],
                                       out_path=out_path, optimizer=lr_opt,
                                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                       steps_per_epoch=len(train_generator),
                                       train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
                    tf.keras.backend.clear_session()
                    return None  # repeat!


if __name__ == '__main__':
    pass
