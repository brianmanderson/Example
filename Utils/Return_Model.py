__author__ = 'Brian M Anderson'
# Created on 5/12/2021
from Deep_Learning.Utils.MyHybridDenseNet.Loading_Pretrained_DenseNet import DenseNet121
import os
from Deep_Learning.Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet, Return_Layer_Functions, return_hollow_layers_dict


def return_model(layers_dict=None, all_trainable=False, weights_path=None):
    model = DenseNet121(include_top=False, classes=2, layers_dict=layers_dict)
    if weights_path is not None and weights_path != 0:
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


if __name__ == '__main__':
    pass
