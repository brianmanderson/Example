__author__ = 'Brian M Anderson'
# Created on 9/1/2020
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
import sys
sys.path.append('..')
from Base_Deeplearning_Code.Models.TF_Keras_Models import base_UNet, ExpandDimension, SqueezeDimension, SqueezeAxes,\
    BreakUpSqueezeDimensions
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export


BASE_WEIGTHS_PATH = ('https://storage.googleapis.com/tensorflow/'
                     'keras-applications/densenet/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')


def dense_block(x, blocks, name):
  """A dense block.

  Arguments:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.

  Returns:
    Output tensor for the block.
  """
  for i in range(blocks):
    x = conv_block(x, 32, name=name + '_block' + str(i + 1))
  return x


def transition_block(x, reduction, name):
    """A transition block.

    Arguments:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.

    Returns:
    output tensor for the block.
    """
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[-1] * reduction), 1,
                      use_bias=False, padding='same', name=name + '_conv')(x)
    just_before = x
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x, just_before


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv', padding='same')(x1)
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
    return x


def squeeze_first2axes_operator(x5d):
    shape = tf.keras.backend.shape(x5d) # get dynamic tensor shape
    x5d = tf.keras.backend.reshape(x5d, [shape[0] * shape[1], shape[2], shape[3], 1])
    return x5d


def return_og_shape(og):
    def break_up_operator(x4d):
        shape = tf.keras.backend.shape(og) # get dynamic tensor shape
        x5d = tf.keras.backend.reshape(x4d, [shape[0], shape[1], shape[2], shape[3], x4d.shape[-1]])
        return x5d
    return break_up_operator


def DenseNet(blocks, include_top=True, weights='imagenet', input_tensor=None, collapse_axis=True,
             model_name='unique', classes=1000, layers_dict=None):
    """Instantiates the DenseNet architecture.
    
    Reference:
    - [Densely Connected Convolutional Networks](
      https://arxiv.org/abs/1608.06993) (CVPR 2017)
    
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    
    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.densenet.preprocess_input` for an example.
    
    Arguments:
    blocks: numbers of building blocks for the four dense layers.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `'channels_last'` data format)
      or `(3, 224, 224)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    
    Returns:
    A `keras.Model` instance.
    
    Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
    """
    # if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization), `imagenet` '
    #                      '(pre-training on ImageNet), '
    #                      'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    input_shape = (None, None, None, 1)
    # Determine proper input shape

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    x = img_input
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    mask = layers.Input(shape=input_shape[:-1] + (1,), name='mask', dtype='int32')
    # x = layers.Lambda(squeeze_first2axes_operator, output_shape=(None, None, None, 1))(x)
    if collapse_axis:
        x = SqueezeAxes()(x)
    x = layers.Concatenate(name='InputConcat')([x, x, x])
    inputs = [img_input, mask]

    encoding = []
    # x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv', padding='Same')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    encoding.append(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x, just_before = transition_block(x, 0.5, name='pool2')
    encoding.append(just_before)
    x = dense_block(x, blocks[1], name='conv3')
    x, just_before = transition_block(x, 0.5, name='pool3')
    encoding.append(just_before)
    x = dense_block(x, blocks[2], name='conv4')
    x, just_before = transition_block(x, 0.5, name='pool4')
    encoding.append(just_before)
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    index = -1
    while encoding:
        index += 1
        x = layers.UpSampling2D(name='Upsampling_Base_{}'.format(index))(x)
        across = encoding.pop()
        filters = int(backend.int_shape(across)[bn_axis])
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3),
                          name='Convolution_{}'.format(index), padding='same')(x)
        x = layers.BatchNormalization(name="BN_{}".format(index))(x)
        x = layers.Activation('relu', name='activation_{}'.format(index))(x)
        x = layers.Add()([x, across])
    x = layers.UpSampling2D(name='Upsampling_Final'.format(index))(x)
    if collapse_axis:
        # x = BreakUpSqueezeDimensions(img_input)(x)
        og_shape = tf.shape(img_input)
        x = tf.reshape(x, [og_shape[0], og_shape[1], og_shape[2], og_shape[3], x.shape[-1]])
        # x = layers.Lambda(return_og_shape(img_input),
        #                   output_shape=(None, None, None, None, int(backend.int_shape(x)[bn_axis])))(x)
    if layers_dict is not None:
        myunet = base_UNet(layers_dict=layers_dict, is_2D=False, explictly_defined=True)
        features_2D = x
        combined_input = layers.Concatenate()([img_input, tf.cast(mask, 'float32')])
        x = layers.Conv3D(32, 5, strides=1, use_bias=False, name='3DConv1', padding='Same')(combined_input)
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='3DConv1/bn')(x)
        x = layers.Activation('relu', name='3DConv1/relu')(x)
        x = layers.Conv3D(32, 3, strides=1, use_bias=False, name='3DConv2', padding='Same')(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='3DConv2/bn')(x)
        x = layers.Activation('relu', name='3DConv2/relu')(x)
        x = x0 = layers.Concatenate()([x, features_2D])
        x = layers.Conv3D(32, 3, strides=2, use_bias=False, name='3DConv3', padding='Same')(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='3DConv3/bn')(x)
        x = layers.Activation('relu', name='3DConv3/relu')(x)

        x = myunet.run_unet(x)

        x = layers.Concatenate()([x, x0])
        x = layers.Conv3D(32, 3, strides=1, use_bias=False, name='3DDecode1', padding='Same')(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='3DDecode1/bn')(x)
        x = layers.Activation('relu', name='3DDecode1/relu')(x)
        x = layers.Conv3D(classes, 1, strides=1, use_bias=False, name='Final_Conv', padding='Same')(x)
        x = layers.Activation('softmax', name='Final_Conv/softmax')(x)
    else:
        x = layers.Conv2D(classes, 1, activation='softmax', padding='same')(x)
    sum_vals_base = tf.where(mask > 0, 0, 1)
    zeros = tf.where(mask > 0, 0, 0)
    zeros = tf.repeat(zeros, repeats=classes-1, axis=-1)
    mask = tf.repeat(mask, repeats=classes, axis=-1)
    sum_vals = tf.concat([sum_vals_base, zeros], axis=-1)
    x = layers.Multiply()([tf.cast(mask, 'float32'), x])
    x = layers.Add()([tf.cast(sum_vals, 'float32'), x])
    if blocks == [6, 12, 24, 16]:
        model_name = 'densenet121'
    elif blocks == [6, 12, 32, 32]:
        model_name = 'densenet169'
    elif blocks == [6, 12, 48, 32]:
        model_name = 'densenet201'
    model = Model(inputs=inputs, outputs=x, name=model_name)
  # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='9d60b8095a5708f2dcce2bca79d332c7')
            elif blocks == [6, 12, 32, 32]:
                weights_path = data_utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='d699b8f76981ab1b30698df4c175e90b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = data_utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='30ee3e1110167f948a6b9946edeeb738')
            elif blocks == [6, 12, 32, 32]:
                weights_path = data_utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = data_utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model

def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000, layers_dict=None):
    """Instantiates the Densenet121 architecture."""
    return DenseNet(blocks=[6, 12, 24, 16], include_top=include_top, weights=weights, input_tensor=input_tensor,
                    classes=classes, layers_dict=layers_dict)
