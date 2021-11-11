__author__ = 'Brian M Anderson'
# Created on 1/30/2020
from Base_Deeplearning_Code.Finding_Optimization_Parameters.History_Plotter import create_excel_from_event, \
    plot_from_excel, os, np, partial, pd
from plotnine import *


make_excel = False
out_path = r'\\mymdafiles\di_data1\Morfeus\bmanderson\Modular_Projects\Liver_Disease_Ablation_Segmentation_Work'
input_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Keras\3D_Atrous_newlrs_livernorm'
excel_out_path = os.path.join(out_path, 'Model_Optimization.xlsx')
if make_excel:
    create_excel_from_event(input_path=input_path,excel_out_path=excel_out_path,
                            names=['Layers','Filters','Max_Filters','Atrous_Blocks','atrous_rate','max_atrous_blocks','min_lr','max_lr','step_size_factor','num_cycles'])

data = pd.read_excel(excel_out_path, engine='openpyxl')
data = data.dropna()
xxx = 1
(ggplot(data) + aes(x='layers',y='val_dice_coef_3D') + facet_wrap('atrous_rate',labeller='label_both') + geom_point(mapping=aes(color='max_filters')) + xlab('Layers') + ylab('Validation Dice') +
 ggtitle('Validation Dice vs Number of Layers and Atrous Rate and Max Filters')+scale_colour_gradient(low='blue',high='red') + theme(text=element_text(size=16)))
(ggplot(data) + aes(x='layers',y='val_loss') + facet_wrap('atrous_rate',labeller='label_both') + geom_point(mapping=aes(color='max_filters')) + xlab('Layers') + ylab('Validation Loss') +
 ggtitle('Validation Loss vs Number of Layers and Atrous Rate and Max Filters')+scale_colour_gradient(low='blue',high='red') + theme(text=element_text(size=16)))
xxx = 1
# criteria_base = lambda x, variable_name, value: np.asarray(list(x[variable_name].values())) == value
# criteria = partial(criteria_base, variable_name='max_filters', value=32)
# plot_from_excel(excel_out_path, variable_name='layers') #,criterias=[criteria]