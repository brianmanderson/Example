__author__ = 'Brian M Anderson'
# Created on 1/17/2020

from Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors import *
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet, Return_Layer_Functions, return_hollow_layers_dict
from Deep_Learning.Utils.MyHybridDenseNet.Loading_Pretrained_DenseNet import DenseNet121
from Deep_Learning.Utils.Return_Paths import return_paths, os
from _collections import OrderedDict
import pandas as pd
import time
from tensorboard.plugins.hparams import api as hp


def return_current_df(run_data, features_list=['layers', 'filters', 'max_filters', 'min_lr', 'max_lr']):
    out_dict = OrderedDict()
    for feature in features_list:
        val = ''
        if feature in run_data:
            val = run_data[feature]
            if type(val) is bool:
                val = int(val)
        out_dict[feature] = [val]
    out_features = [i for i in out_dict.keys()]
    return pd.DataFrame(out_dict), out_features


def return_pandas_df(excel_path, features_list=['layers','filters','max_filters','min_lr','max_lr']):
    if not os.path.exists(excel_path):
        out_dict = OrderedDict()
        out_dict['Trial_ID'] = []
        for key in features_list:
            out_dict[key] = []
        df = pd.DataFrame(out_dict)
        df.to_excel(excel_path, index=0)
    else:
        df = pd.read_excel(excel_path, engine='openpyxl')
    return df


def return_hyper_parameters():
    HP_NUM_LAYERS = hp.HParam('layers', hp.Discrete([2, 3, 4]))
    HP_FILTERS = hp.HParam('filters', hp.Discrete([8, 16]))
    HP_MAX_FILTERS = hp.HParam('max_filters', hp.Discrete([32, 64, 128]))
    hp_dict = {'layers':HP_NUM_LAYERS, 'filters':HP_FILTERS, 'max_filters':HP_MAX_FILTERS}
    return hp_dict


def return_hparams(run_data, features_list, excluded_keys=['iteration','save']):
    hparams = None
    for layer_key in features_list:
        break_out = False
        for exclude in excluded_keys:
            if layer_key.lower().find(exclude) != -1:
                break_out = True
        if break_out:
            continue
        if layer_key in run_data.keys():
            if hparams is None:
                hparams = OrderedDict()
            hparams[hp.HParam(layer_key, hp.Discrete([run_data[layer_key]]))] = run_data[layer_key]
    return hparams


def return_best_dictionary(base_dict):
    dictionary = [
        base_dict(max_conv_blocks=4, layers=2, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.75e-3),
        # base_dict(max_conv_blocks=4, layers=3, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
        #           min_lr=4e-7, max_lr=1.75e-3),
                  ]
    return dictionary


def return_dictionary(base_dict):
    dictionary = [
        base_dict(max_conv_blocks=4, layers=2, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.75e-3),
        base_dict(max_conv_blocks=4, layers=3, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.75e-3),
        base_dict(max_conv_blocks=4, layers=4, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.2e-3),
        base_dict(max_conv_blocks=4, layers=4, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=3e-4),

        base_dict(max_conv_blocks=6, layers=2, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.6e-3),
        base_dict(max_conv_blocks=6, layers=3, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.6e-3),
        base_dict(max_conv_blocks=6, layers=3, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.6e-3),
        base_dict(max_conv_blocks=6, layers=4, num_conv_blocks=3, conv_lambda=1, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=1.2e-3),
        base_dict(max_conv_blocks=6, layers=4, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=3e-4),

        base_dict(max_conv_blocks=8, layers=3, num_conv_blocks=3, conv_lambda=2, filters=32, max_filters=128,
                  min_lr=4e-7, max_lr=3e-4)
                  ]
    return dictionary


def return_dictionary_densenet3D(base_dict, all_trainable=False):
    if all_trainable:
        dictionary = [
            base_dict(layers=2, num_conv_blocks=2, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=1e-7,
                      max_lr=2e-4),
            base_dict(layers=2, num_conv_blocks=2, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=1e-7,
                      max_lr=7e-4),
            ]
    else:
        dictionary = [
            base_dict(layers=2, num_conv_blocks=2, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=1.6e-3),
            base_dict(layers=2, num_conv_blocks=2, conv_lambda=4, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=4e-2),
            base_dict(layers=2, num_conv_blocks=2, conv_lambda=4, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=1e-2),

            base_dict(layers=2, num_conv_blocks=4, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=2e-2),
            base_dict(layers=2, num_conv_blocks=4, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=2e-3),

            base_dict(layers=2, num_conv_blocks=4, conv_lambda=4, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=4e-3),

            base_dict(layers=3, num_conv_blocks=2, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=2e-3),
            base_dict(layers=3, num_conv_blocks=2, conv_lambda=4, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=2e-3),
            base_dict(layers=3, num_conv_blocks=2, conv_lambda=4, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=2e-2),

            base_dict(layers=3, num_conv_blocks=4, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=1e-1),
            base_dict(layers=3, num_conv_blocks=4, conv_lambda=2, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=2e-3),
            base_dict(layers=3, num_conv_blocks=4, conv_lambda=4, filters=32, max_conv_blocks=12, min_lr=2e-6,
                      max_lr=2e-3),
            ]
    return dictionary


def return_dictionary_dense(base_dict, run_best=False, is_2D=False):
    if is_2D:
        dictionary = [
            base_dict(layers=2, max_conv_blocks=4, filters=8, num_conv_blocks=2, conv_lambda=1, growth_rate=4,
                      min_lr=1e-6, max_lr=1e-3)
            ]
        return dictionary
    dictionary = [
        base_dict(layers=2, max_conv_blocks=4, filters=8, num_conv_blocks=2, conv_lambda=0, growth_rate=4,
                  min_lr=6e-7, max_lr=1e-3),
        base_dict(layers=2, max_conv_blocks=4, filters=8, num_conv_blocks=2, conv_lambda=1, growth_rate=4,
                  min_lr=6e-7, max_lr=2e-3),
        base_dict(layers=2, max_conv_blocks=4, filters=12, num_conv_blocks=2, conv_lambda=0, growth_rate=4,
                  min_lr=7e-7, max_lr=4e-3),
        base_dict(layers=2, max_conv_blocks=4, filters=12, num_conv_blocks=2, conv_lambda=1, growth_rate=4,
                  min_lr=7e-7, max_lr=3e-4),

        base_dict(layers=3, max_conv_blocks=4, filters=8, num_conv_blocks=2, conv_lambda=0, growth_rate=4,
                  min_lr=7e-7, max_lr=1e-3),
        base_dict(layers=3, max_conv_blocks=4, filters=8, num_conv_blocks=2, conv_lambda=1, growth_rate=4,
                  min_lr=7e-7, max_lr=8e-4),

        base_dict(layers=3, max_conv_blocks=4, filters=12, num_conv_blocks=2, conv_lambda=0, growth_rate=4,
                  min_lr=7e-7, max_lr=1e-3),
        base_dict(layers=3, max_conv_blocks=4, filters=12, num_conv_blocks=2, conv_lambda=1, growth_rate=4,
                  min_lr=7e-7, max_lr=3e-4),
                  ]
    if run_best:
        dictionary = [
            base_dict(layers=2, max_conv_blocks=4, filters=8, num_conv_blocks=2, conv_lambda=0, growth_rate=4,
                      min_lr=6e-7, max_lr=8e-3)
                      ]
    dictionary = [
        base_dict(layers=2, max_conv_blocks=4, filters=8, num_conv_blocks=2, conv_lambda=1, growth_rate=0,
                  min_lr=2e-6, max_lr=2e-2),
    ]
    return dictionary


def get_atrous_layers_dict(layers=1, filters=16, max_filters=np.inf, num_conv_blocks=2, conv_lambda=0, bn_before_activation=True, **kwargs):
    lc = Return_Layer_Functions(kernel=(3,3,3),strides=(1,1,1),padding='same',batch_norm=True,
                                pooling_type='Max', pool_size=(2,2,2), bn_before_activation=bn_before_activation)
    dfkw = {'padding':'same','batch_norm':True, 'activation':'elu'}
    layers_dict = {}
    for layer in range(layers):
        encoding = []
        for _ in range(num_conv_blocks):
            encoding.append(lc.atrous_layer(filters, **dfkw))
        if layer != 0:
            encoding = [lc.residual_layer(encoding, **dfkw)]
        layers_dict['Layer_{}'.format(layer)] = {'Encoding':encoding}
        if filters < max_filters:
            filters = int(filters*2)
        num_conv_blocks += conv_lambda
    final_steps = [lc.convolution_layer(32, **dfkw),
                   lc.convolution_layer(2, batch_norm=False, activation='softmax')]
    layers_dict['Final_Steps'] = final_steps
    return layers_dict


def get_layers_dict_dense(layers=1, filters=12, growth_rate=6, max_filters=np.inf, conv_lambda=0, num_conv_blocks=2, max_conv_blocks=4, num_classes=2,**kwargs):
    lc = Return_Layer_Functions(kernel=(3,3,3),strides=(1,1,1),padding='same',batch_norm=True,
                                pooling_type='Max', pool_size=(2,2,2), bn_before_activation=False)

    block = lc.convolution_layer
    start = [block(filters,out_name='start', batch_norm=False, activation=None)]

    layers_dict = return_hollow_layers_dict(layers)
    pool = (2, 2, 2)
    previous_name = 'start'
    final_filters = None
    for layer in range(layers - 1):
        if layer == 0:
            layers_dict['Layer_' + str(layer)]['Encoding'] = start
        else:
            layers_dict['Layer_' + str(layer)]['Encoding'] = []
        encoding = []
        names = [previous_name]
        for i in range(num_conv_blocks):
            name = 'Layer_{}_Conv_Encoding_{}'.format(layer, i)
            names.append(name)
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, kernel=(1,1,1), batch_norm=True, activation='elu'),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
            encoding += [lc.concat_layer(names)]
            names = names[:]
            filters += growth_rate
        layers_dict['Layer_' + str(layer)]['Encoding'] += encoding
        previous_name = 'Layer_{}_Down'.format(layer)
        up_name = 'Layer_{}_Up'.format(layer)
        layers_dict['Layer_' + str(layer)]['Pooling']['Decoding'] = [lc.upsampling_layer(pool_size=pool),
                                                                     lc.convolution_layer(filters, activation=None,
                                                                                          batch_norm=False,
                                                                                          out_name=up_name)]
        if filters < max_filters:
            filters = int(filters*2)
        layers_dict['Layer_' + str(layer)]['Pooling']['Encoding'] = lc.convolution_layer(filters, strides=(2,2,2),
                                                                                         activation=None,
                                                                                         batch_norm=False,
                                                                                         out_name=previous_name)
        layers_dict['Layer_' + str(layer)]['Decoding'] = []
        encoding = []
        names = [up_name]
        for i in range(num_conv_blocks):
            name = 'Layer_{}_Conv_Decoding_{}'.format(layer, i)
            names.append(name)
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, kernel=(1,1,1), batch_norm=True, activation='elu'),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
            encoding += [lc.concat_layer(names)]
            names = names[:]
            filters += growth_rate
        if layer == 0:
            final_filters = filters
        layers_dict['Layer_' + str(layer)]['Decoding'] = encoding
        num_conv_blocks += conv_lambda
        num_conv_blocks = min([num_conv_blocks, max_conv_blocks])
    encoding = []
    names = [previous_name]
    for i in range(num_conv_blocks):
        name = 'Base_{}'.format(i)
        names.append(name)
        encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                     block(filters, kernel=(1, 1, 1), batch_norm=True, activation='elu'),
                     block(filters, batch_norm=False, activation=None, out_name=name)]
        encoding += [lc.concat_layer(names)]
        names = names[:]
        filters += growth_rate
    if final_filters is None:
        final_filters = filters
    final_steps = [lc.convolution_layer(final_filters, batch_norm=True, kernel=(1,1,1),activation='elu'),
                   lc.convolution_layer(num_classes, batch_norm=False, activation='softmax')]
    layers_dict['Final_Steps'] = final_steps
    layers_dict['Base'] = encoding
    return layers_dict


def get_layers_dict_dense_new(layers=1, filters=12, growth_rate=6, conv_lambda=0, num_conv_blocks=2, max_conv_blocks=4,
                              num_classes=2, pool=(2, 2, 2), kernel=(3, 3, 3), squeeze_kernel=(1, 1, 1),
                              max_filters=np.inf, **kwargs):
    lc = Return_Layer_Functions(kernel=kernel, strides=squeeze_kernel, padding='same', batch_norm=True,
                                pooling_type='Max', pool_size=pool, bn_before_activation=False)
    block = lc.convolution_layer
    start = [block(filters, out_name='start', batch_norm=False, activation=None)]

    layers_dict = return_hollow_layers_dict(layers)
    previous_name = 'start'
    encoding_names = []
    layers_encoding = []
    num_blocks = []
    for layer in range(layers - 1):
        num_blocks.append(num_conv_blocks)
        layers_encoding.append(layer)
        if layer == 0:
            layers_dict['Layer_' + str(layer)]['Encoding'] = start
        else:
            layers_dict['Layer_' + str(layer)]['Encoding'] = []
        encoding = []
        names = [previous_name]
        for i in range(num_conv_blocks):
            name = 'Layer_{}_Conv_Encoding_{}'.format(layer, i)
            names.append(name)
            if i != 0:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            else:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            encoding += [lc.concat_layer(names)]
            names = names[:]
            filters += growth_rate
            filters = min([filters, max_filters])
        filters *= 2
        encoding_names.append(names)
        layers_dict['Layer_' + str(layer)]['Encoding'] += encoding
        previous_name = 'Layer_{}_Down'.format(layer)
        layers_dict['Layer_' + str(layer)]['Pooling']['Encoding'] = [
            lc.activation_layer('elu'), lc.batch_norm_layer(),
            lc.convolution_layer(filters, strides=pool, activation=None, batch_norm=False, out_name=previous_name)
        ]
        num_conv_blocks += conv_lambda
        num_conv_blocks = min([num_conv_blocks, max_conv_blocks])
    # We want the filter number to still grow by growth_factor, so add in the decoding side later...
    encoding = []
    names = [previous_name]
    for i in range(num_conv_blocks):
        name = 'Base_{}'.format(i)
        names.append(name)
        if i != 0:
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
        else:
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
        encoding += [lc.concat_layer(names)]
        names = names[:]
        filters += growth_rate
        filters = min([filters, max_filters])

    layers_dict['Base'] = encoding
    for layer in layers_encoding[::-1]:
        filters //= 2
        num_conv_blocks = num_blocks[layer]
        up_name = 'Layer_{}_Up'.format(layer)
        layers_dict['Layer_' + str(layer)]['Pooling']['Decoding'] = [
            lc.activation_layer('elu'), lc.batch_norm_layer(),
            lc.upsampling_layer(pool_size=pool),
            lc.convolution_layer(filters, activation=None, batch_norm=False, out_name=up_name)
        ]
        layers_dict['Layer_' + str(layer)]['Decoding'] = []
        encoding = []
        names = [up_name]
        names += encoding_names[layer]
        for i in range(num_conv_blocks):
            name = 'Layer_{}_Conv_Decoding_{}'.format(layer, i)
            names.append(name)
            if i != 0:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            else:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            encoding += [lc.concat_layer(names)]
            names = names[:]
            filters += growth_rate
            filters = min([filters, max_filters])
        layers_dict['Layer_' + str(layer)]['Decoding'] = encoding
    final_steps = [lc.activation_layer('elu'), lc.batch_norm_layer(),
                   block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                   lc.convolution_layer(num_classes, batch_norm=False, activation='softmax')]
    layers_dict['Final_Steps'] = final_steps
    return layers_dict

def get_layers_dict_dense_HNet(layers=1, filters=12, growth_rate=0, conv_lambda=0, num_conv_blocks=2, max_conv_blocks=4,
                               pool=(2, 2, 2), kernel=(3, 3, 3), squeeze_kernel=(1, 1, 1),
                               max_filters=np.inf, **kwargs):
    lc = Return_Layer_Functions(kernel=kernel, strides=squeeze_kernel, padding='same', batch_norm=True,
                                pooling_type='Max', pool_size=pool, bn_before_activation=False)
    block = lc.convolution_layer
    start = []
    names = []
    layers_dict = return_hollow_layers_dict(layers)
    previous_name = 'start'
    encoding_names = []
    layers_encoding = []
    num_blocks = []
    for layer in range(layers - 1):
        num_blocks.append(num_conv_blocks)
        layers_encoding.append(layer)
        if layer == 0:
            layers_dict['Layer_' + str(layer)]['Encoding'] = start
        else:
            layers_dict['Layer_' + str(layer)]['Encoding'] = []
        encoding = []
        for i in range(num_conv_blocks):
            name = 'Layer_{}_Conv_Encoding_{}'.format(layer, i)
            names.append(name)
            if i != 0:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            else:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            if len(names) > 1:
                encoding += [lc.concat_layer(names)]
            names = names[:]
            filters += growth_rate
            filters = min([filters, max_filters])
        filters *= 2
        encoding_names.append(names)
        layers_dict['Layer_' + str(layer)]['Encoding'] += encoding
        previous_name = 'Layer_{}_Down'.format(layer)
        names = [previous_name]
        layers_dict['Layer_' + str(layer)]['Pooling']['Encoding'] = [
            lc.activation_layer('elu'), lc.batch_norm_layer(),
            lc.convolution_layer(filters, strides=pool, activation=None, batch_norm=False, out_name=previous_name)
        ]
        num_conv_blocks += conv_lambda
        num_conv_blocks = min([num_conv_blocks, max_conv_blocks])
    # We want the filter number to still grow by growth_factor, so add in the decoding side later...
    encoding = []
    names = [previous_name]
    for i in range(num_conv_blocks):
        name = 'Base_{}'.format(i)
        names.append(name)
        if i != 0:
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
        else:
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
        encoding += [lc.concat_layer(names)]
        names = names[:]
        filters += growth_rate
        filters = min([filters, max_filters])

    layers_dict['Base'] = encoding
    for layer in layers_encoding[::-1]:
        filters //= 2
        up_name = 'Layer_{}_Up'.format(layer)
        layers_dict['Layer_' + str(layer)]['Pooling']['Decoding'] = [
            lc.activation_layer('elu'), lc.batch_norm_layer(),
            lc.upsampling_layer(pool_size=pool),
            lc.convolution_layer(filters, activation='relu', batch_norm=True, out_name=up_name)
        ]
        encoding = []
        names = [up_name, encoding_names[layer][-1]]
        encoding += [lc.concat_layer(names),
                     block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu')
                     ]
        layers_dict['Layer_' + str(layer)]['Decoding'] = encoding
    final_steps = [lc.upsampling_layer(pool_size=pool),
                   lc.convolution_layer(filters, activation='relu', batch_norm=True, out_name='Upsampling_Final_Steps')]
    layers_dict['Final_Steps'] = final_steps
    return layers_dict

def get_layers_dict_dense_less_decode(layers=1, filters=12, growth_rate=6, conv_lambda=0, num_conv_blocks=2,
                                      max_conv_blocks=4, num_classes=2, pool=(2, 2, 2), kernel=(3, 3, 3),
                                      squeeze_kernel=(1, 1, 1), max_filters=np.inf, **kwargs):
    lc = Return_Layer_Functions(kernel=kernel, strides=squeeze_kernel, padding='same', batch_norm=True,
                                pooling_type='Max', pool_size=pool, bn_before_activation=False)
    block = lc.convolution_layer
    start = [block(filters, out_name='start', batch_norm=False, activation=None)]

    layers_dict = return_hollow_layers_dict(layers)
    previous_name = 'start'
    encoding_names = []
    layers_encoding = []
    num_blocks = []
    for layer in range(layers - 1):
        num_blocks.append(num_conv_blocks)
        layers_encoding.append(layer)
        if layer == 0:
            layers_dict['Layer_' + str(layer)]['Encoding'] = start
        else:
            layers_dict['Layer_' + str(layer)]['Encoding'] = []
        encoding = []
        names = [previous_name]
        for i in range(num_conv_blocks):
            name = 'Layer_{}_Conv_Encoding_{}'.format(layer, i)
            names.append(name)
            if i != 0:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            else:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            encoding += [lc.concat_layer(names)]
            names = names[:]
            filters += growth_rate
            filters = min([filters, max_filters])
        filters *= 2
        encoding_names.append(names)
        layers_dict['Layer_' + str(layer)]['Encoding'] += encoding
        previous_name = 'Layer_{}_Down'.format(layer)
        layers_dict['Layer_' + str(layer)]['Pooling']['Encoding'] = [
            lc.activation_layer('elu'), lc.batch_norm_layer(),
            lc.convolution_layer(filters, strides=pool, activation=None, batch_norm=False, out_name=previous_name)
        ]
        num_conv_blocks += conv_lambda
        num_conv_blocks = min([num_conv_blocks, max_conv_blocks])
    # We want the filter number to still grow by growth_factor, so add in the decoding side later...
    encoding = []
    names = [previous_name]
    for i in range(num_conv_blocks):
        name = 'Base_{}'.format(i)
        names.append(name)
        if i != 0:
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
        else:
            encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                         block(filters, batch_norm=False, activation=None, out_name=name)]
        encoding += [lc.concat_layer(names)]
        names = names[:]
        filters += growth_rate
        filters = min([filters, max_filters])

    layers_dict['Base'] = encoding
    for layer in layers_encoding[::-1]:
        filters //= 2
        num_conv_blocks = num_blocks[layer]
        up_name = 'Layer_{}_Up'.format(layer)
        layers_dict['Layer_' + str(layer)]['Pooling']['Decoding'] = [
            lc.activation_layer('elu'), lc.batch_norm_layer(),
            lc.upsampling_layer(pool_size=pool),
            lc.convolution_layer(filters, activation='elu', batch_norm=True, out_name=up_name)
        ]
        layers_dict['Layer_' + str(layer)]['Decoding'] = []
        encoding = []
        names = [up_name]
        names += encoding_names[layer]
        for i in range(num_conv_blocks):
            name = 'Layer_{}_Conv_Decoding_{}'.format(layer, i)
            names.append(name)
            if i != 0:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            else:
                encoding += [lc.activation_layer('elu'), lc.batch_norm_layer(),
                             block(filters, batch_norm=False, activation=None, out_name=name)]
            encoding += [lc.concat_layer(names)]
            names = names[:]
            filters += growth_rate
            filters = min([filters, max_filters])
        layers_dict['Layer_' + str(layer)]['Decoding'] = encoding
    final_steps = [lc.activation_layer('elu'), lc.batch_norm_layer(),
                   block(filters, kernel=squeeze_kernel, batch_norm=True, activation='elu'),
                   lc.convolution_layer(num_classes, batch_norm=False, activation='softmax')]
    layers_dict['Final_Steps'] = final_steps
    return layers_dict


def return_model(layers_dict=None, is_2D=False, densenet=False, all_trainable=False, weights_path=None):
    image_size = (None, None, None, 1)
    if is_2D:
        image_size = (None, None, 1)
    if not densenet:
        model = my_UNet(layers_dict=layers_dict, image_size=image_size,
                        mask_output=True, explictly_defined=True, is_2D=is_2D).created_model
    else:
        model = DenseNet121(include_top=False, classes=2, layers_dict=layers_dict)
        if weights_path is not None:
            print('Loading weights from {}'.format(weights_path))
            if not os.path.exists(weights_path):
                model.load_weights(weights_path.replace('.h5', '.cpkt'))
                model.save_weights(weights_path)
            else:
                model.load_weights(weights_path.replace('.cpkt', '.h5'), by_name=True)
        if not all_trainable:
            freeze_name = 'Upsampling'
            if layers_dict is not None:
                freeze_name = 'Upsampling_Final'
            trainable = False
            for index, layer in enumerate(model.layers):
                if layer.name.find(freeze_name) == 0:
                    trainable = True
                model.layers[index].trainable = trainable
    return model


def return_base_dict(step_size_factor=10, save_a_model=False,optimizer='Adam'):
    base_dict = lambda min_lr, max_lr, layers, num_conv_blocks, max_conv_blocks, conv_lambda, filters, max_filters: \
        OrderedDict({'atrous':False, 'layers': layers,'num_conv_blocks':num_conv_blocks, 'max_conv_blocks':max_conv_blocks,
                     'conv_lambda':conv_lambda, 'filters':filters, 'max_filters':max_filters,
                     'Save_Model':save_a_model,'Optimizer':optimizer, 'min_lr':min_lr,
                     'max_lr':max_lr, 'step_size_factor': step_size_factor
                     })
    return base_dict


def return_base_dict_dense(step_size_factor=10, save_a_model=False):
    base_dict = lambda min_lr, max_lr, layers, num_conv_blocks, max_conv_blocks, conv_lambda, filters, growth_rate: \
        OrderedDict({'layers': layers,'num_conv_blocks':num_conv_blocks, 'max_conv_blocks':max_conv_blocks,
                     'conv_lambda':conv_lambda, 'filters':filters, 'growth_rate':growth_rate,
                     'Save_Model':save_a_model,'min_lr':min_lr,
                     'max_lr':max_lr, 'step_size_factor': step_size_factor
                     })
    return base_dict


def return_base_dict_dense3D():
    base_dict = lambda min_lr, max_lr, filters, layers, num_conv_blocks, max_conv_blocks, conv_lambda: \
        OrderedDict({'layers': layers,'num_conv_blocks':num_conv_blocks, 'max_conv_blocks':9999,
                     'conv_lambda':conv_lambda, 'filters':filters, 'min_lr':min_lr, 'max_lr':max_lr})
    return base_dict


def return_generators(batch_size=16, wanted_keys={'inputs':['image','mask'],'outputs':['annotation']},path_lead='Records_1mm',
                      add='', is_test=False, cache=True, validation_name='',cache_add='', flip=False,
                      change_background=False, threshold=False, threshold_val=10, evaluation=False, train_path=None,
                      validation_path=None):
    base_path, morfeus_drive = return_paths()
    if not os.path.exists(base_path):
        print('{} does not exist'.format(base_path))
    if train_path is None:
        train_path = [os.path.join(base_path, path_lead, 'Train{}_Records'.format(add))]
    if validation_path is None:
        validation_path = [os.path.join(base_path, path_lead, 'Validation_Records{}'.format(validation_name))]
    ext = 'Validation'
    if evaluation:
        ext += '_whole'
    if is_test:
        validation_path = [os.path.join(base_path, path_lead, 'Test_Records{}'.format(validation_name))]

    train_generator = DataGeneratorClass(record_paths=train_path, debug=False)
    validation_generator = DataGeneratorClass(record_paths=validation_path, in_parallel=True)
    train_processors, validation_processors = [], []
    base_processors = [
        Expand_Dimensions(axis=-1, on_images=True, on_annotations=False),
                        ]
    train_processors += base_processors
    validation_processors += base_processors
    train_processors += [
        Ensure_Image_Proportions(image_rows=64, image_cols=64),
        Return_Add_Mult_Disease(change_background=change_background),
    ]
    train_processors += [
        Cast_Data({'annotation': 'float16', 'mask': 'int32'})]
    if not evaluation and cache:
        train_processors += [
        {'cache': os.path.join(base_path,'cache', 'Train{}{}'.format(add, cache_add))}
    ]
    validation_processors += [
        Return_Add_Mult_Disease(change_background=change_background),
        Cast_Data({'annotation': 'float16', 'mask': 'int32'})]
    if cache:
        validation_processors += [
        {'cache': os.path.join(base_path,'cache','{}{}{}'.format(ext,add,cache_add))}
        ]
    if threshold:
        train_processors += [
            Threshold_Images(lower_bound=-threshold_val, upper_bound=threshold_val)
        ]
        validation_processors += [
            Threshold_Images(lower_bound=-threshold_val, upper_bound=threshold_val)
        ]
    if flip:
        train_processors += [
        Flip_Images(keys=['image', 'mask', 'annotation'],
                    flip_lr=True, flip_up_down=True, flip_3D_together=True,flip_z=True)
        ]
    train_processors += [
        Return_Outputs(wanted_keys),
        {'shuffle': len(train_generator)}]
    validation_processors += [
        Return_Outputs(wanted_keys)]
    if batch_size != 0:
        train_processors += [{'batch': batch_size}]
        validation_processors += [{'batch': 1}]
    train_processors += [{'repeat'}]
    validation_processors += [{'repeat'}]
    train_generator.compile_data_set(image_processors=train_processors, debug=False)
    validation_generator.compile_data_set(image_processors=validation_processors, debug=False)
    start = time.time()
    generators = [validation_generator]
    if not evaluation:
        generators += [train_generator]
        for generator in generators: #
            data_set = iter(generator.data_set)
            for _ in range(len(generator)):
                x, y = next(data_set)
                print(x[0].shape)
    print(time.time()-start)
    #     print(data[1][0].shape)
    # data = next(data_set)
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    kernel = (3, 3)
    batch_size = 12
    squeeze_kernel = (1, 1)

    add = '_32'
    path_desc = 'TF_LR_2D_DenseNetMultiBatch'
    excel_file_name = 'parameters_list_by_trial_id_DenseNetMultibatch.xlsx'
    model_name = 'DenseNetNewMultiBatch'
    cache_add = ''
    path_lead = 'Records_1mm'
    base_path, morfeus_drive, train_generator, validation_generator = return_generators(
        batch_size=batch_size, add=add, threshold_val=10, change_background=False,
        cache_add=cache_add, path_lead=path_lead, validation_name='', cache=False)
    pass
