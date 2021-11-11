__author__ = 'Brian M Anderson'
# Created on 12/8/2020
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.History_Plotter_TF2 import np, \
    iterate_paths_add_to_dictionary
from .Return_Paths import return_paths
import pandas as pd
import os


def add_metrics_to_excel():
    base_dictionary = {}
    base_path, morfeus_drive, excel_path = return_paths()
    df = pd.read_excel(excel_path, engine='openpyxl')
    not_filled_df = df.loc[
        (~pd.isnull(df['Iteration']))
        & (pd.isnull(df['epoch_loss']))
        ]
    df.set_index('Model_Index', inplace=True)
    path_base = os.path.join(morfeus_drive, 'Tensorflow')
    rewrite = False
    for index in not_filled_df.index.values:
        model_index = not_filled_df['Model_Index'][index]
        path = os.path.join(path_base, 'Model_Index_{}'.format(model_index))
        if not os.path.exists(path):
            continue
        rewrite = True
        iterate_paths_add_to_dictionary(path=path, all_dictionaries=base_dictionary, fraction_start=0.1,
                                        weight_smoothing=0.8,
                                        metric_name_and_criteria={'epoch_loss': np.min})
    if rewrite:
        out_dictionary = {'Model_Index': [], 'epoch_loss': []}
        for key in base_dictionary.keys():
            out_dictionary['Model_Index'].append(int(key.split('_')[-1]))
            out_dictionary['epoch_loss'].append(base_dictionary[key]['epoch_loss'])
        new_df = pd.DataFrame(out_dictionary)
        new_df.set_index('Model_Index', inplace=True)
        df.update(new_df)
        df = df.reset_index()
        df.to_excel(excel_path, index=0)
    return None


if __name__ == '__main__':
    pass
