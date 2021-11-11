__author__ = 'Brian M Anderson'
# Created on 5/29/2020

import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

from Return_Train_Validation_Generators_TF2 import return_dictionary_densenet3D, get_layers_dict_dense_HNet, return_generators, return_base_dict_dense3D, return_model, plot_scroll_Image, os
from Run_Model_TF2 import SparseCategoricalMeanDSC
import numpy as np
import SimpleITK as sitk
# from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras as tfk
import tensorflow as tf

# from HDenseUNet.denseunet import DenseUNet
# from MyHybridDenseNet.Loading_Pretrained_DenseNet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet121
add = '_16'
path_desc='TF_LR_2D_Dense_1mm_new'
model_name = 'DenseNet2'
cache_add = ''
# x,y = next(iter(train_generator.data_set))
xxx = 1
base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=5, add=add,cache_add=cache_add,
                                                                                    flip=True, change_background=False,
                                                                                    threshold=True, threshold_val=10,
                                                                                    path_lead='Records', validation_name='_64',
                                                                                    cache=False, wanted_keys={'inputs':['image','mask'],'outputs':['annotation']})
base_dict = return_base_dict_dense3D()
overall_dictionary = return_dictionary_densenet3D(base_dict, all_trainable=False)
run_data = overall_dictionary[0]
layers_dict = get_layers_dict_dense_HNet(**run_data)
x1,y1 = next(iter(validation_generator.data_set))
model_path = r'H:\Liver_Disease_Ablation\Keras\DenseNetNewMultiBatch\Models\Trial_ID_13\Model'

print('saving model')
model = return_model(layers_dict=layers_dict, is_2D=True, densenet=True, all_trainable=False)
model.load_weights(r'H:\Liver_Disease_Ablation\Keras\DenseNetNewMultiBatch\Models\Trial_ID_13\cp-0101.h5', by_name=True)
if not os.path.exists(model_path):
    model.save(model_path)
# else:
#     model = tf.keras.models.load_model(model_path)
#     print('loaded model')
pred = model.predict(x1)
model.compile(tfk.optimizers.Adam(),
              loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tfk.metrics.SparseCategoricalAccuracy(), SparseCategoricalMeanDSC(num_classes=2)])
# model.load_weights(os.path.join('.', 'Test.hdf5'))
# xx, yy = next(iter(validation_generator.data_set))
model.fit(train_generator.data_set, epochs=10,steps_per_epoch=len(train_generator),
          validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
          validation_freq=1, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=r'H:\Test', profile_batch=0, write_graph=True)])
# model.save_weights(os.path.join('.', 'Test.hdf5'))
# model.save(os.path.join('.', 'Test'), save_format='h5')
# xxx = 1
# model = DenseUNet(reduction=0.5)
# model.load_weights(r'H:\Liver_Disease_Ablation\densenet161_weights_tf.h5', by_name=True)
# out_path = r'D:\test_model\test'
# layers_dict = get_layers_dict_dense_less_decode(layers=4, filters=8, growth_rate=0, num_conv_blocks=2, conv_lambda=0)
# model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1),
#                 mask_output=True).created_model
# k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
# k.set_model(model)
# k.on_train_begin()