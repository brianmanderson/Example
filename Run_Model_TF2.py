__author__ = 'Brian M Anderson'

# Created on 1/18/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
import tensorflow as tf
from Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR, SparseCategoricalMeanDSC
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Data_Generators.Return_Paths import Path_Return_Class
from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import CyclicLR
from Return_Train_Validation_Generators_TF2 import return_generators, return_base_dict_dense, return_base_dict_dense3D, \
    return_hparams, return_dictionary_densenet3D, return_pandas_df, return_current_df, np, return_paths, \
    return_best_dictionary, \
    get_layers_dict_dense_new, return_dictionary_dense, return_model, get_layers_dict_dense_HNet
from tensorboard.plugins.hparams.keras import Callback


def run_model(batch_size, add, cache_add, flip, change_background, threshold, threshold_val, path_lead, validation_name,
              trial_id, min_lr=1e-4, max_lr=1e-2, layers_dict=None, epochs=1000, base_reduce_factor=100,
              reduction_factor=100,
              paths_class=None, step_size_factor=5, morfeus_drive='', base_path='', run_best=False,
              skip_cyclic_lr=False, scale_mode='linear_cycle', optimizer='SGD', hparams=None, kernel=(3, 3, 3),
              all_trainable=False, densenet=False, weights_path=None, include_images=True, **kwargs):
    _, _, train_generator, validation_generator = return_generators(batch_size=batch_size, add=add, cache_add=cache_add,
                                                                    flip=flip, change_background=change_background,
                                                                    threshold=threshold, threshold_val=threshold_val,
                                                                    path_lead=path_lead,
                                                                    validation_name=validation_name)
    step_size = len(train_generator)
    x, y = next(iter(train_generator.data_set))
    xx, yy = next(iter(validation_generator.data_set))
    print(x[0].shape)
    print(xx[0].shape)
    if not os.path.exists(morfeus_drive):
        print('Morf wrong')
        return None
    if not os.path.exists(base_path):
        print('base wrong')
        return None

    Model_val = return_model(layers_dict, is_2D=kernel == (3, 3), all_trainable=all_trainable, densenet=densenet,
                             weights_path=weights_path)
    model_path_out = paths_class.model_path_out
    tensorboard_output = paths_class.tensorboard_path_out
    if optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD()
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam()
    # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    print('Learning rate is {}'.format(min_lr))

    if os.listdir(tensorboard_output):
        print('already done')
        return None
    # checkpoint_path = os.path.join(model_path_out,'cp-best.ckpt')
    image_frequency = 10
    patience = 30
    checkpoint_path = os.path.join(model_path_out, 'cp-{epoch:04d}.cpkt')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                 save_freq='epoch', save_best_only=False, save_weights_only=True, mode='min',
                                 verbose=1)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_output, profile_batch=0,
                                                 write_graph=True)  # profile_batch='300,401',
    lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=step_size, step_size_factor=step_size_factor,
                     mode='triangular2', pre_cycle=0, base_reduce_factor=base_reduce_factor, scale_mode=scale_mode,
                     step_size_factor_scale=lambda x: x + 2, reduction_factor=reduction_factor)
    callbacks = [tensorboard]
    if include_images:
        add_images = Add_Images_and_LR(log_dir=tensorboard_output, validation_data=validation_generator.data_set,
                                       number_of_images=len(validation_generator), add_images=True,
                                       image_frequency=image_frequency,
                                       threshold_x=True)
        callbacks += [add_images]
    if hparams is not None:
        hp_callback = Callback(tensorboard_output, hparams=hparams, trial_id='Trial_ID:{}'.format(trial_id))
        callbacks += [hp_callback]
    if not skip_cyclic_lr:
        callbacks += [lrate]
    callbacks += [checkpoint]
    # if not run_best:
    #     callbacks += [EarlyStopping(patience=patience, verbose=1)]
    print('\n\n\n\nRunning {}\n\n\n\n'.format(tensorboard_output))
    Model_val.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), SparseCategoricalMeanDSC(num_classes=2)])
    Model_val.fit(train_generator.data_set, epochs=epochs, steps_per_epoch=len(train_generator),
                  validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
                  validation_freq=1, callbacks=callbacks)
    Model_val.save(os.path.join(model_path_out, 'final_model.h5'))
    tf.keras.backend.clear_session()
    return None


def compare_base_current(data_frame, current_run_df, features_list):
    current_array = current_run_df[features_list].values
    base_array = data_frame[features_list].values
    if np.any(base_array) and np.max([np.min(i == current_array) for i in base_array]):
        return True
    return False


def train_model(epochs=None, save_a_model=False, model_name='3D_Fully_Atrous',
                run_best=False, debug=False, add='', dense=False, cache_add='_1mm', kernel=(3, 3, 3),
                squeeze_kernel=(1, 1, 1), is_2D=False, batch_size=8, change_background=True,
                excel_file_name='parameters_list_by_trial_id_Dense.xlsx'):
    optimizers = ['Adam']
    pool = (2, 2, 2)
    if is_2D:
        pool = (2, 2)
    concat = True
    if run_best:
        save_a_model = True
    step_size_factor = 6
    threshold = True
    for iteration in [0, 1, 2]:
        for flip in [True]:
            for threshold_val in [10]:
                for optimizer in optimizers:
                    base_path, morfeus_drive = return_paths()
                    # base_dict = return_base_dict(step_size_factor=step_size_factor,
                    #                              save_a_model=save_a_model, optimizer=optimizer)
                    base_dict = return_base_dict_dense(step_size_factor=step_size_factor, save_a_model=save_a_model)
                    if run_best:
                        if dense:
                            overall_dictionary = return_dictionary_dense(base_dict, run_best=run_best, is_2D=is_2D)
                        else:
                            overall_dictionary = return_best_dictionary(base_dict)
                    else:
                        # overall_dictionary = return_dictionary(base_dict)
                        overall_dictionary = return_dictionary_dense(base_dict, is_2D=is_2D)
                    overall_dictionary = np.asarray(overall_dictionary)
                    perm = np.arange(len(overall_dictionary))
                    np.random.shuffle(perm)
                    overall_dictionary = overall_dictionary[perm]
                    for run_data in overall_dictionary:
                        run_data['percentile_normed'] = True
                        run_data['sampling'] = 1
                        run_data['mirror_max'] = False
                        run_data['Model_Style'] = 'new'
                        run_data['concat'] = concat
                        run_data['flipped'] = flip
                        run_data['change_background'] = change_background
                        run_data['threshold'] = threshold
                        run_data['threshold_val'] = threshold_val
                        tf.random.set_seed(iteration)
                        run_data['batch_size'] = batch_size
                        excel_path = os.path.join(morfeus_drive, excel_file_name)
                        print(base_path)
                        run_data['Iteration'] = iteration
                        run_data['Trial_ID'] = 0
                        data_frame = return_pandas_df(excel_path, features_list=list(run_data.keys()))
                        trial_id = 0
                        while trial_id in data_frame['Trial_ID'].values:
                            trial_id += 1
                        run_data['Trial_ID'] = trial_id
                        current_run_df, features_list = return_current_df(run_data, features_list=data_frame.columns)
                        if compare_base_current(data_frame=data_frame, current_run_df=current_run_df,
                                                features_list=[i for i in data_frame.columns if i != 'Trial_ID']):
                            print('Already done')
                            continue
                        print(current_run_df)
                        data_frame = data_frame.append(current_run_df, ignore_index=True)
                        data_frame.to_excel(excel_path, index=0)
                        _, _, train_generator, validation_generator = return_generators(batch_size=batch_size, add=add,
                                                                                        cache_add=cache_add,
                                                                                        flip=flip,
                                                                                        change_background=change_background,
                                                                                        threshold=threshold,
                                                                                        threshold_val=threshold_val)
                        step_size = len(train_generator)
                        hparams = return_hparams(run_data, features_list=features_list, excluded_keys=[])

                        layers_dict = get_layers_dict_dense_new(**run_data, kernel=kernel,
                                                                squeeze_kernel=squeeze_kernel, pool=pool)
                        paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive,
                                                        save_model=save_a_model,
                                                        is_keras_model=False)
                        paths_class.define_model_things(model_name, 'Trial_ID_{}'.format(trial_id))
                        tensorboard_output = paths_class.tensorboard_path_out
                        print(tensorboard_output)
                        if os.listdir(tensorboard_output):
                            print('already done')
                            continue
                        run_model(trial_id=str(trial_id), layers_dict=layers_dict, train_generator=train_generator,
                                  step_size=step_size, optimizer=optimizer,
                                  validation_generator=validation_generator, run_best=run_best, kernel=kernel,
                                  squeeze_kernel=squeeze_kernel,
                                  paths_class=paths_class, morfeus_drive=morfeus_drive, hparams=hparams,
                                  base_path=base_path, epochs=epochs, **run_data)
                        return None  # break out!


def train_DenseNet(epochs=None, save_a_model=False, model_name='3D_Fully_Atrous',
                   run_best=False, add='', cache_add='_1mm', batch_size=0,
                   change_background=False, excel_file_name='parameters_list_by_trial_id_DenseNetMultibatch.xlsx',
                   all_trainable=False, path_lead='', validation_name='', weights_path=None, layers_dict=None):
    optimizers = ['Adam']
    concat = True
    if run_best:
        save_a_model = True
    threshold = True
    run_data = {}
    for iteration in [0, 1]:
        for base_reduce_factor, reduction_factor in zip([10, 100], [10, 100]):
            for max_lr in [1e-3, 1e-2, 1e-1]:
                for flip in [True]:
                    for threshold_val in [10]:
                        for optimizer in optimizers:
                            run_data['base_reduce_factor'] = base_reduce_factor
                            run_data['reduction_factor'] = reduction_factor
                            base_path, morfeus_drive = return_paths()
                            if not all_trainable and layers_dict is None:
                                run_data['min_lr'] = 2e-6
                                run_data['max_lr'] = 8e-4
                            elif layers_dict is not None:
                                run_data['min_lr'] = 4e-7
                                run_data['max_lr'] = 3e-4
                            else:
                                run_data['min_lr'] = 1e-6
                                run_data['max_lr'] = max_lr
                            run_data['percentile_normed'] = True
                            run_data['sampling'] = 1
                            run_data['mirror_max'] = False
                            run_data['Model_Style'] = 'DenseNet2D'
                            run_data['concat'] = concat
                            run_data['all_trainable'] = all_trainable
                            run_data['flip'] = flip
                            run_data['change_background'] = change_background
                            run_data['threshold'] = threshold
                            run_data['threshold_val'] = threshold_val
                            run_data['batch_size'] = batch_size
                            run_data['densenet'] = True
                            excel_path = os.path.join(morfeus_drive, excel_file_name)
                            print(base_path)
                            run_data['Iteration'] = iteration
                            run_data['Trial_ID'] = 0
                            data_frame = return_pandas_df(excel_path, features_list=list(run_data.keys()))
                            trial_id = 0
                            while trial_id in data_frame['Trial_ID'].values:
                                trial_id += 1
                            run_data['Trial_ID'] = trial_id
                            current_run_df, features_list = return_current_df(run_data,
                                                                              features_list=data_frame.columns)
                            if compare_base_current(data_frame=data_frame, current_run_df=current_run_df,
                                                    features_list=[i for i in data_frame.columns if i != 'Trial_ID']):
                                print('Already done')
                                continue
                            print(current_run_df)
                            data_frame = data_frame.append(current_run_df, ignore_index=True)
                            data_frame.to_excel(excel_path, index=0)
                            hparams = return_hparams(run_data, features_list=features_list, excluded_keys=[])
                            hparams = None

                            paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive,
                                                            save_model=save_a_model,
                                                            is_keras_model=False)
                            paths_class.define_model_things(model_name, 'Trial_ID_{}'.format(trial_id))
                            tensorboard_output = paths_class.tensorboard_path_out
                            print(tensorboard_output)
                            if os.listdir(tensorboard_output):
                                print('already done')
                                continue
                            run_model(validation_name=validation_name, add=add, cache_add=cache_add,
                                      trial_id=str(trial_id),
                                      layers_dict=layers_dict, optimizer=optimizer, include_images=False,
                                      run_best=run_best, path_lead=path_lead,
                                      paths_class=paths_class, morfeus_drive=morfeus_drive, hparams=hparams,
                                      base_path=base_path, epochs=epochs, weights_path=weights_path, **run_data)
                            return None  # break out!


def train_DenseNet3D(epochs=None, save_a_model=False, model_name='3D_Fully_Atrous',
                     run_best=False, add='', cache_add='_1mm', batch_size=0,
                     change_background=False, excel_file_name='parameters_list_by_trial_id_DenseNet3D.xlsx',
                     all_trainable=False, path_lead='', validation_name='', weights_path=None):
    optimizers = ['Adam']
    concat = True
    if run_best:
        save_a_model = True
    threshold = True
    base_reduce_factor = 100
    reduction_factor = 100
    for iteration in [0, 1, 2]:
        for flip in [True]:
            for threshold_val in [10]:
                for optimizer in optimizers:
                    base_path, morfeus_drive = return_paths()
                    base_dict = return_base_dict_dense3D()
                    overall_dictionary = return_dictionary_densenet3D(base_dict, all_trainable=all_trainable)
                    overall_dictionary = np.asarray(overall_dictionary)
                    perm = np.arange(len(overall_dictionary))
                    np.random.shuffle(perm)
                    overall_dictionary = overall_dictionary[perm]
                    for run_data in overall_dictionary:
                        run_data['base_reduce_factor'] = base_reduce_factor
                        run_data['reduction_factor'] = reduction_factor
                        run_data['percentile_normed'] = True
                        run_data['sampling'] = 1
                        run_data['flip'] = flip
                        run_data['mirror_max'] = False
                        run_data['Model_Style'] = 'DenseNet3D3layer'
                        run_data['concat'] = concat
                        run_data['all_trainable'] = all_trainable
                        run_data['flipped'] = flip
                        run_data['change_background'] = change_background
                        run_data['threshold'] = threshold
                        run_data['threshold_val'] = threshold_val
                        run_data['batch_size'] = batch_size
                        run_data['densenet'] = True
                        excel_path = os.path.join(morfeus_drive, excel_file_name)
                        layers_dict = get_layers_dict_dense_HNet(**run_data)
                        print(base_path)
                        run_data['Iteration'] = iteration
                        run_data['Trial_ID'] = 0
                        data_frame = return_pandas_df(excel_path, features_list=list(run_data.keys()))
                        trial_id = 0
                        tensorboard_path = os.path.join(morfeus_drive, 'Keras', model_name, 'Tensorboard')
                        while trial_id in data_frame['Trial_ID'].values or 'Trial_ID_{}'.format(trial_id) in os.listdir(tensorboard_path):
                            trial_id += 1
                        run_data['Trial_ID'] = trial_id
                        current_run_df, features_list = return_current_df(run_data, features_list=data_frame.columns)
                        if compare_base_current(data_frame=data_frame, current_run_df=current_run_df,
                                                features_list=[i for i in data_frame.columns if i != 'Trial_ID']):
                            print('Already done')
                            continue
                        print(current_run_df)
                        data_frame = data_frame.append(current_run_df, ignore_index=True)
                        data_frame.to_excel(excel_path, index=0)
                        hparams = return_hparams(run_data, features_list=features_list, excluded_keys=[])
                        paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive,
                                                        save_model=save_a_model, is_keras_model=False)
                        paths_class.define_model_things(model_name, 'Trial_ID_{}'.format(trial_id))
                        tensorboard_output = paths_class.tensorboard_path_out
                        print(tensorboard_output)
                        if os.listdir(tensorboard_output):
                            print('already done')
                            continue
                        run_model(validation_name=validation_name, add=add, cache_add=cache_add, trial_id=str(trial_id),
                                  layers_dict=layers_dict, optimizer=optimizer, include_images=True,
                                  run_best=run_best, path_lead=path_lead,
                                  paths_class=paths_class, morfeus_drive=morfeus_drive, hparams=hparams,
                                  base_path=base_path, epochs=epochs, weights_path=weights_path, **run_data)
                        return None  # break out!


if __name__ == '__main__':
    pass
