__author__ = 'Brian M Anderson'
# Created on 5/24/2021
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths
from plotnine import *
import pandas as pd


def view_results_with_r():
    base_path, morfeus_drive, excel_path = return_paths()
    df = pd.read_excel(excel_path, engine='openpyxl')
    df = df[
        (~pd.isnull(df['epoch_loss']))
            ]
    for variable in ['step_factor', 'min_lr', 'max_lr']:
        xxx = 1
        (ggplot(df) + aes(x='{}'.format(variable), y='epoch_loss') + geom_point(mapping=aes(color='epoch_loss'))
         + facet_wrap('batch_size', labeller='label_both') +
         xlab('{}'.format(variable)) + ylab('Validation Loss') + scale_y_log10() + scale_x_log10()
         + scale_colour_gradient(low='blue', high='red') + ggtitle('Title'))
        xxx = 1