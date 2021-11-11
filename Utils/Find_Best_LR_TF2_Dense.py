__author__ = 'Brian M Anderson'

# Created on 4/26/2020
import tensorflow as tf
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import is_df_within_another
from tensorflow.keras.callbacks import TensorBoard
from .Return_Model import return_model
from .Return_Generators import return_generators, return_paths
import os
import pandas as pd
import numpy as np


def return_model_parameters(excel_path, iteration, out_path):
    base_df = pd.read_excel(excel_path, engine='openpyxl')
    base_df.set_index('Model_Index')
    potentially_not_run = base_df.loc[pd.isnull(base_df.Iteration) & pd.isnull(base_df.min_lr)]
    indexes_for_not_run = potentially_not_run.index.values
    for index in indexes_for_not_run:
        run_df = base_df.loc[[index]]
        model_parameters = run_df.squeeze().to_dict()
        model_index = run_df.loc[index, 'Model_Index']
        model_out_path = os.path.join(out_path, 'Model_Index_{}'.format(model_index),
                                      '{}_Iteration'.format(iteration))
        if os.path.exists(model_out_path):
            continue
        os.makedirs(model_out_path)
        for key in model_parameters.keys():
            value = model_parameters[key]
            if type(model_parameters[key]) is np.int64 or value == 0:
                model_parameters[key] = int(model_parameters[key])
            elif type(model_parameters[key]) is np.float64:
                model_parameters[key] = float(model_parameters[key])
        return model_parameters, model_out_path
    return None, None


def find_best_lr_DenseNet(weights_path=None):
    min_lr = 1e-7
    max_lr = 1
    base_path, morfeus_drive, excel_path = return_paths()
    out_path = os.path.join(morfeus_drive, 'Learning_Rates')
    for iteration in [0]:
        model_parameters, model_out_path = return_model_parameters(excel_path=excel_path, iteration=iteration,
                                                                   out_path=out_path)
        if model_parameters is None:
            continue
        is_2D = model_parameters['is_2D']
        batch_size = model_parameters['batch_size']
        all_trainable = model_parameters['all_trainable']
        train_generator, validation_generator = return_generators(is_2D=is_2D, cache=True, batch=batch_size)
        if model_parameters['layers'] == 0:
            layers_dict = None
        else:
            layers_dict = model_parameters
        model = return_model(layers_dict=layers_dict, weights_path=weights_path, all_trainable=all_trainable)
        k = TensorBoard(log_dir=model_out_path, profile_batch=0, write_graph=True)
        k.set_model(model)
        k.on_train_begin()
        lr_opt = tf.keras.optimizers.Adam
        print(model_out_path)
        LearningRateFinder(epochs=10, model=model, metrics=['categorical_accuracy'],
                           out_path=model_out_path, optimizer=lr_opt,
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           steps_per_epoch=10000,
                           train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
        tf.keras.backend.clear_session()
        return None  # repeat!


if __name__ == '__main__':
    pass
