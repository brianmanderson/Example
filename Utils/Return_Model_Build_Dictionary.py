__author__ = 'Brian M Anderson'
# Created on 5/12/2021
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Models.TF_Keras_Models import Return_Layer_Functions, return_hollow_layers_dict

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


if __name__ == '__main__':
    pass
