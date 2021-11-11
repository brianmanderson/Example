__author__ = 'Brian M Anderson'
# Created on 1/30/2020
from Base_Deeplearning_Code.Finding_Optimization_Parameters.History_Plotter_TF2 import create_excel_from_event, \
    plot_from_excel, os, pd
from plotnine import *


def main(make_excel = False, input_path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Disease_Ablation_Segmentation_Work\Keras\3D_Fully_Atrous'):
    out_path = '.'
    excel_out_path = os.path.join(out_path, 'Model_Optimization.xlsx')
    if make_excel or not os.path.exists(excel_out_path):
        create_excel_from_event(input_path=input_path,excel_out_path=excel_out_path, names=['Trial_ID'])

    data = pd.read_excel(excel_out_path, engine='openpyxl')
    data = data.dropna()
    xxx = 1
    (ggplot(data) + aes(x='layers',y='val_dice_coef_3D') + facet_wrap('filters',labeller='label_both') + geom_point(mapping=aes(color='max_filters')) + xlab('Layers') + ylab('Validation Dice') +
     ggtitle('Validation Dice vs Number of Layers, Filters, and Max Filters')+scale_colour_gradient(low='blue',high='red'))
    (ggplot(data) + aes(x='layers',y='val_loss') + facet_wrap('filters',labeller='label_both') + geom_point(mapping=aes(color='max_filters')) + xlab('Layers') + ylab('Validation Dice') +
     ggtitle('Validation Loss vs Number of Layers, Filters, and Max Filters')+scale_colour_gradient(low='blue',high='red')+scale_y_log10())
    xxx = 1
    # criteria_base = lambda x, variable_name, value: np.asarray(list(x[variable_name].values())) == value
    # criteria = partial(criteria_base, variable_name='max_filters', value=32)
    # plot_from_excel(excel_out_path, variable_name='layers') #,criterias=[criteria]


if __name__ == '__main__':
    pass
