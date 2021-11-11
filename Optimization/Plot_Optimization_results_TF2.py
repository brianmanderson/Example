__author__ = 'Brian M Anderson'
# Created on 1/30/2020
from plotnine import *
import os
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.History_Plotter_TF2 import create_excel_from_event, combine_hparamxlsx_and_metricxlsx
import pandas as pd


def main():
    out_path = '.'
    excel_out_path = os.path.join(out_path, 'hparams_table.xlsx')
    create_excel_from_event(input_path=r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Disease_Ablation_Segmentation_Work\Keras\TF2_3D_Fully_Atrous_Variable_Cube_Training\Tensorboard',
                            excel_out_path=excel_out_path)
    hparameters_path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Disease_Ablation_Segmentation_Work\parameters_list_by_trial_id.xlsx'
    combined_xlsx = os.path.join('.','combined.xlsx')
    combine_hparamxlsx_and_metricxlsx(hparameters_path, excel_out_path, combined_xlsx)
    data = pd.read_excel(os.path.join('.','combined.xlsx'), engine='openpyxl')
    data = data.dropna()
    xxx = 1
    (ggplot(data) + aes(x='layers', y='epoch_loss') + facet_wrap('num_conv_blocks',
                                                                     labeller='label_both') + geom_point(
        mapping=aes(color='conv_lambda')) + xlab('layers') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Number of Conv Blocks, and Conv Lambda') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))
    (ggplot(data) + aes(x='layers', y='log_epoch_loss') + facet_wrap('conv_lambda', labeller='label_both') + geom_point(
        mapping=aes(color='num_conv_blocks')) + xlab('Layers') + ylab('Validation Loss') +
     ggtitle('Validation Loss vs Number of Layers, Filters, and Max Filters') + scale_colour_gradient(low='blue',
                                                                                                      high='red'))


if __name__ == '__main__':
    pass
