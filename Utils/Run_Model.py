__author__ = 'Brian M Anderson'
# Created on 5/14/2021
from Deep_Learning.Utils.Return_Generators import return_generators, return_paths
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import \
    is_df_within_another, return_hparams
import os
from Deep_Learning.Utils.Return_Model import return_model
import pandas as pd
import tensorflow as tf
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR, MeanDSC
from Deep_Learning.Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import SGDRScheduler
from tensorflow.keras import metrics
import os
from tensorboard.plugins.hparams.keras import Callback


def run_model(model, train_generator, validation_generator, min_lr, max_lr, model_path, tensorboard_path, trial_id,
              label_smoothing=0., hparams=None, step_factor=8, epochs=120):
    step_factor = int(step_factor)
    epochs = int(epochs)
    label_smoothing = float(label_smoothing)
    min_lr, max_lr = float(min_lr), float(max_lr)
    checkpoint_path = os.path.join(model_path, 'cp-best.cpkt')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min', verbose=1,
                                                    save_freq='epoch', save_best_only=True, save_weights_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path,# profile_batch='400,601',
                                                 write_graph=True)
    lrate = SGDRScheduler(min_lr=min_lr, max_lr=max_lr, steps_per_epoch=len(train_generator), cycle_length=step_factor,
                          lr_decay=0.5, mult_factor=1, gentle_start_epochs=0, gentle_fraction=1.0)
    add_lr = Add_Images_and_LR(log_dir=tensorboard_path, add_images=True, validation_data=validation_generator.data_set,
                               number_of_images=len(validation_generator), image_frequency=10)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=True, mode='min')
    callbacks = [tensorboard, lrate, add_lr]
    # if epochs < 9000:
    callbacks += [early_stop]
    if hparams is not None:
        hp_callback = Callback(tensorboard_path, hparams=hparams, trial_id='Trial_ID:{}'.format(trial_id))
        callbacks += [hp_callback]
    callbacks += [checkpoint]
    METRICS = [
        metrics.CategoricalAccuracy(),
        MeanDSC(num_classes=2)
    ]
    print('\n\n\n\nRunning {}\n\n\n\n'.format(tensorboard_path))
    model.compile(tf.keras.optimizers.Adam(learning_rate=min_lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), metrics=METRICS)
    model.fit(train_generator.data_set, epochs=epochs, steps_per_epoch=len(train_generator),
              validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
              validation_freq=5, callbacks=callbacks)
    model.save(os.path.join(model_path, 'final_model.h5'))
    tf.keras.backend.clear_session()
    return None


def write_to_excel(excel_path, iterations):
    base_df = pd.read_excel(excel_path, engine='openpyxl')
    base_df.set_index('Model_Index')
    potentially_not_run = base_df.loc[pd.isnull(base_df.Iteration) & ~pd.isnull(base_df.min_lr)]
    indexes_for_not_run = potentially_not_run.index.values
    guess_index = 0
    rewrite = False
    for index in indexes_for_not_run:
        for iteration in iterations:
            run_df = base_df.loc[[index]]
            run_df.at[index, 'Iteration'] = iteration
            compare_list = ('min_lr', 'max_lr', 'Iteration', 'batch_size', 'layers', 'filters', 'growth_rate',
                            'conv_lambda', 'num_conv_blocks', 'is_2D', 'all_trainable')
            contained = is_df_within_another(data_frame=base_df, current_run_df=run_df, features_list=compare_list)
            if not contained:
                while guess_index in base_df['Model_Index'].values:
                    guess_index += 1
                run_df.at[index, 'reference'] = run_df.loc[index, 'Model_Index']
                run_df.at[index, 'Model_Index'] = guess_index
                run_df.set_index('Model_Index')
                base_df = base_df.append(run_df, ignore_index=True)
                rewrite = True
    if rewrite:
        base_df.to_excel(excel_path, index=0)
    return rewrite


def run_2d_model():
    tf.random.set_seed(3141)
    epochs = 100001
    base_path, morfeus_drive, excel_path = return_paths()
    iterations = [0]
    if base_path.startswith('H'):
        rewrite = write_to_excel(excel_path=excel_path, iterations=iterations)
        if rewrite:
            return None

    base_df = pd.read_excel(excel_path, engine='openpyxl')
    base_df.set_index('Model_Index')
    potentially_not_run = base_df.loc[~pd.isnull(base_df.Iteration)
                                      & pd.isnull(base_df['epoch_loss'])
                                      ]
    indexes_for_not_run = potentially_not_run.index.values
    np.random.shuffle(indexes_for_not_run)
    for index in indexes_for_not_run:
        run_df = base_df.loc[[index]]
        model_index = int(run_df.loc[index, 'Model_Index'])
        tensorboard_path = os.path.join(morfeus_drive, 'Tensorflow', 'Model_Index_{}'.format(model_index))
        if os.path.exists(tensorboard_path):
            continue
        model_parameters = run_df.squeeze().to_dict()
        for key in model_parameters.keys():
            if type(model_parameters[key]) is np.int64:
                model_parameters[key] = int(model_parameters[key])
            elif type(model_parameters[key]) is np.float64:
                model_parameters[key] = float(model_parameters[key])
        os.makedirs(tensorboard_path)
        features_list = ('min_lr', 'max_lr', 'Iteration', 'batch_size', 'layers', 'filters', 'growth_rate',
                         'conv_lambda', 'num_conv_blocks', 'is_2D', 'all_trainable')
        train_generator, validation_generator = return_generators(is_2D=model_parameters['is_2D'],
                                                                  batch_size=model_parameters['batch_size'],
                                                                  cache=True)
        if model_parameters['is_2D']:
            layers_dict = None
        else:
            layers_dict = model_parameters
        model = return_model(all_trainable=model_parameters['all_trainable'], layers_dict=layers_dict,
                             weights_path=model_parameters['weights_path'])

        model_path = os.path.join(base_path, 'Models', 'Model_Index_{}'.format(model_index))

        print('Saving model to {}\ntensorboard at {}'.format(model_path, tensorboard_path))
        hparams = return_hparams(model_parameters, features_list=features_list, excluded_keys=[])
        run_model(model=model, train_generator=train_generator, validation_generator=validation_generator,
                  min_lr=model_parameters['min_lr'], max_lr=model_parameters['max_lr'], model_path=model_path,
                  tensorboard_path=tensorboard_path, trial_id=model_index, hparams=hparams,
                  step_factor=model_parameters['step_factor'], epochs=epochs,
                  label_smoothing=model_parameters['label_smoothing'])
        return False
    return True


if __name__ == '__main__':
    pass
