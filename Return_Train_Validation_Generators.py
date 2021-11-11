__author__ = 'Brian M Anderson'
# Created on 1/17/2020

from Base_Deeplearning_Code.Data_Generators.Keras_Generators import Data_Generator_Class
from Deep_Learning.Utils.Return_Paths import return_paths
from _collections import OrderedDict


def return_base_dict(step_size_factor=10, step_size_add=3, save_a_model=False,sgd_opt=False):
    base_dict = lambda min_lr, max_lr, layers, filters, max_filters: \
        OrderedDict({'Architecture':{'model_name':'','layers': layers,'atrous_blocks': 2,
                                     'filters':filters, 'max_filters':max_filters},
                     'Hyper_Parameters':{'Save_Model':save_a_model,'Final':True,'SGD_Opt':sgd_opt, 'min_lr':min_lr,
                                         'max_lr':max_lr, 'step_size_factor': step_size_factor,
                                         'step_size_add':step_size_add,
                                         }
                     })
    return base_dict


def get_layers_dict_atrous(layers=1, filters=16, atrous_blocks=2, max_atrous_blocks=2, max_filters=np.inf,
                           atrous_rate=2, **kwargs):
    activation = {'activation':'elu'}
    layers_dict = {}
    atrous_block = lambda x, y, z: {'atrous': {'channels': x, 'atrous_rate': y, 'activations': z}}
    residual_block = lambda x: {'residual':x}
    for layer in range(layers-1):
        encoding = []
        for i in range(atrous_rate):
            encoding += [residual_block([atrous_block(filters, atrous_rate, ['elu' for _ in range(atrous_rate)])])]
            encoding += [activation]
        if filters < max_filters:
            filters = int(filters * 2)
        layers_dict['Layer_' + str(layer)] = {'Encoding': encoding}
        if atrous_blocks < max_atrous_blocks:
            atrous_blocks = int(atrous_blocks * 2)
    encoding = []
    for i in range(atrous_rate):
        encoding += [residual_block([atrous_block(filters, atrous_rate, ['elu' for _ in range(atrous_rate)])])]
        encoding += [activation]
    layers_dict['Base'] = encoding
    layers_dict['Final_Steps'] = [{'convolution':{'channels': 16, 'kernel': (3, 3, 3), 'strides': (1, 1, 1), 'activation': 'elu'}},
                                  {'convolution':{'channels': 2, 'kernel': (1, 1, 1), 'strides': (1, 1, 1), 'activation': 'softmax'}}]
    return layers_dict

def return_dictionary(base_dict):
    dictionary = {
        1: [
            base_dict(1e-5, 1e-2, 8, 16, 1),
            base_dict(1e-5, 3e-3, 16, 16, 1),
            base_dict(3e-6, 2e-3, 32, 32, 1),
            base_dict(1e-5, 2e-3, 8, 16, 2),
            base_dict(2e-6, 1e-3, 16, 16, 2),
            base_dict(2e-6, 2e-4, 32, 32, 2),
            base_dict(1e-5, 2e-4, 8, 16, 3),
            base_dict(2e-6, 1.7e-4, 16, 16, 3),
            base_dict(1e-6, 6e-5, 32, 32, 3)
        ],
        2: [
            base_dict(6e-6, 2e-4, 16, 16, 1),
            base_dict(1e-6, 1e-3, 32, 32, 1),
            base_dict(2e-6, 1e-4, 16, 16, 2),
            base_dict(1e-6, 2.5e-4, 32, 32, 2),
            base_dict(1.7e-6, 5e-5, 16, 16, 3),
            base_dict(1e-6, 4e-5, 32, 32, 3)
        ],
        3: [
            base_dict(2e-6, 1e-3, 16, 16, 1),
            base_dict(1e-6, 1.5e-4, 32, 32, 1),
            base_dict(1e-6, 1e-4, 16, 16, 2),
            base_dict(5e-7, 7e-5, 32, 32, 2),
            base_dict(1e-6, 8e-4, 16, 16, 3),
            base_dict(1e-6, 2.5e-5, 32, 32, 3)
        ],
        4: [
            base_dict(5e-6, 2e-4, 16, 16, 1),
            base_dict(1e-6, 1.5e-4, 32, 32, 1),
            base_dict(1.8e-6, 4e-4, 16, 16, 2),
            base_dict(1e-6, 2e-4, 32, 32, 2),
            base_dict(1e-6, 5e-5, 16, 16, 3)
        ],
        5: [
            base_dict(1.5e-6, 2e-4, 16, 16, 1),
            base_dict(2e-7, 2e-4, 32, 32, 1),
            base_dict(2e-6, 1.5e-4, 16, 16, 2),
            base_dict(2e-7, 5e-5, 32, 32, 2),
            base_dict(1e-6, 1e-4, 16, 16, 3)
        ],
        6: [
            base_dict(2e-6, 6e-5, 16, 16, 1),
            base_dict(2e-7, 3e-5, 32, 32, 1),
            base_dict(1e-6, 8e-5, 16, 16, 2),
            base_dict(2e-7, 1e-4, 32, 32, 2)
        ],
        7: [
            base_dict(8e-7, 1e-4, 16, 16, 1),
            base_dict(2e-7, 7e-6, 32, 32, 1),
            base_dict(1e-6, 1e-4, 16, 16, 2),
            base_dict(2e-7, 2e-4, 32, 32, 2)
        ],
        8: [
            base_dict(1e-6, 1e-4, 16, 16, 1),
            base_dict(6e-7, 2e-5, 32, 32, 1),
            base_dict(6e-7, 4e-5, 16, 16, 2)
        ]
    }
    return dictionary


def return_dictionary_cube(base_dict):
    # base_dict = lambda min_lr, max_lr, layers, filters, max_filters
    dictionary = [
        base_dict(1e-5, 1e-3, 1, 8, 16),
        base_dict(1e-5, 1e-3, 1, 8, 32),
        base_dict(1e-5, 8e-4, 1, 16, 16),
        base_dict(8e-6, 8e-4, 1, 16, 32),

        base_dict(8e-6, 6e-4, 2, 8, 16),
        base_dict(1e-6, 8e-4, 2, 8, 32),
        base_dict(1e-6, 3e-4, 2, 16, 16),
        base_dict(1e-6, 2e-4, 2, 16, 32),

        base_dict(5e-6, 2e-4, 3, 8, 16),
        base_dict(1e-6, 6e-5, 3, 8, 32),
        base_dict(4e-6, 1e-4, 3, 16, 16),
        base_dict(1e-6, 8e-5, 3, 16, 32),

        base_dict(1e-6, 1.3e-4, 4, 8, 16),
        base_dict(1e-6, 2e-4, 4, 8, 32),
        base_dict(2e-6, 1e-4, 4, 16, 16),
        base_dict(2e-6, 1e-4, 4, 16, 32),

        base_dict(1e-6, 1e-4, 5, 8, 16),
        base_dict(1e-6, 1e-4, 5, 8, 32),
        base_dict(1e-6, 1e-4, 5, 16, 16),
        base_dict(1e-6, 2e-5, 5, 16, 32)
    ]
    return dictionary


def return_dictionary_new_training(base_dict):
    # base_dict = lambda min_lr, max_lr, layers, filters, max_filters
    dictionary = [
        base_dict(2e-6, 3e-4, 1, 8, 16),
        base_dict(2e-6, 2e-4, 1, 8, 32),
        base_dict(2e-6, 2e-4, 1, 16, 16),
        base_dict(2e-6, 2e-4, 1, 16, 32),

        base_dict(2e-6, 1e-4, 2, 8, 16),
        base_dict(1e-6, 1e-4, 2, 8, 32),
        base_dict(1e-6, 1.5e-4, 2, 16, 16),
        base_dict(5e-7, 1e-4, 2, 16, 32),

        base_dict(8e-7, 1.5e-4, 3, 8, 16),
        base_dict(8e-7, 8e-5, 3, 8, 32),
        base_dict(5e-7, 1e-4, 3, 16, 16),
        base_dict(5e-7, 8e-5, 3, 16, 32),

        base_dict(4e-7, 1e-5, 4, 8, 16),
        base_dict(3e-7, 1e-5, 4, 8, 32),
        base_dict(5e-7, 2e-5, 4, 16, 16),
        base_dict(1e-7, 2e-5, 4, 16, 32),

        base_dict(1e-7, 3e-5, 5, 8, 16),
        base_dict(2e-7, 2e-5, 5, 8, 32),
        base_dict(6e-7, 2e-5, 5, 16, 16),
        base_dict(2e-7, 1e-5, 5, 16, 32)
    ]
    return dictionary


def return_dictionary_best(base_dict, sgd=False):
    dictionary = [
        base_dict(1e-6, 2e-4, 4, 8, 32)
    ]
    if sgd:
        dictionary = [
            base_dict(1e-5, 1.5e-3, 4, 8, 32)
        ]
    return dictionary


def return_dictionary_best_4layer(base_dict):
    dictionary = {
        4: [
            base_dict(1e-6, 2e-4, 32, 32, 2)
            # base_dict(1e-3, 0, 32, 32, 2)
        ]
    }
    return dictionary


def return_generators(get_mean_std=False, liver_norm=True,num_patients=1,
                      cube_size=None, path_extension='Single_Images3D', return_test=False):
    base_path, morfeus_drive = return_paths()
    if not os.path.exists(base_path):
        print('{} does not exist'.format(base_path))
    paths = [os.path.join(base_path, 'Train', path_extension)]
    paths_validation_generator = [os.path.join(base_path, 'Validation', path_extension)]
    paths_test_generator = [os.path.join(base_path, 'Test', path_extension)]
    num_classes = 3
    mean_val = 67
    std_val = 36
    expansion = 16
    if get_mean_std:
        mean_val = 0
        std_val = 1
    if liver_norm:
        normalize = Normalize_to_Liver() # was (0.1 to 0.75)
    else:
        normalize = Normalize_Images(mean_val=mean_val,std_val=std_val)
    image_processors_train = [normalize,Ensure_Image_Proportions(512, 512),
                              Annotations_To_Categorical(num_of_classes=num_classes)
                              ]
    if cube_size is not None:
        image_processors_train += [
                              Pull_Cube_sitk(annotation_index=2, max_cubes=cube_size[0], z_images=cube_size[1],
                                             rows=cube_size[2], cols=cube_size[3], min_volume=0, min_voxels=300,
                                             max_volume=np.inf, max_voxels=1350000)]
    image_processors_train += [
                              # Threshold_Images(lower_bound=lower_bound, upper_bound=upper_bound,
                              #                  inverse_image=inverse_images),
                              Mask_Pred_Within_Annotation(return_mask=True, liver_box=True, mask_image=False,
                                                          remove_liver_layer_indexes=(0,2))
                              ]
    image_processors_test = [normalize,
                             Ensure_Image_Proportions(512, 512),
                             # Threshold_Images(lower_bound=lower_bound, upper_bound=upper_bound,
                             #                  inverse_image=inverse_images),
                             Annotations_To_Categorical(num_of_classes=num_classes),
                             Clip_Images(annotations_index=(1,2)),
                             Mask_Pred_Within_Annotation(return_mask=True, liver_box=True, mask_image=False,
                                                         remove_liver_layer_indexes=(0, 2))
                             ]
    train_generator = Data_Generator_Class(by_patient=True,num_patients=num_patients, whole_patient=True, shuffle=False,
                                           data_paths=paths, expansion=expansion, wanted_indexes=[1],
                                           image_processors=image_processors_train)
    validation_generator = Data_Generator_Class(by_patient=True,num_patients=1, whole_patient=True, shuffle=False,
                                                data_paths=paths_validation_generator, wanted_indexes=[1],expansion=expansion,
                                                image_processors=image_processors_test)
    while False:
        for i in range(1,2):
            print(i)
            x, y = train_generator.__getitem__(i)
            print(x[0].shape[0])
            xxx = 1
    if return_test:
        test_generator = Data_Generator_Class(by_patient=True,num_patients=1, whole_patient=True, shuffle=False,
                                              data_paths=paths_test_generator, expansion=expansion,
                                              image_processors=image_processors_test)
        return base_path, morfeus_drive, 0, test_generator
    # x,y = validation_generator.__getitem__(0)
    if get_mean_std:
        livers = []
        diseases = []
        for i in range(len(train_generator)):
            print(i)
            # print(train_generator.generator.image_list)
            x, y = train_generator.__getitem__(i)
            data = x[y[..., 2] == 1][..., 0]
            livers.append(np.mean(x[y[...,1]==1]))
            diseases.append(np.mean(x[y[...,2]==1]))
            # print(np.mean(data))
            if i == 0:
                output = data
            else:
                output = np.append(output, data, axis=0)
        print(np.mean(output, axis=0))
        print(np.std(output, axis=0))
        print(np.median(output, axis=0))
        fid = open(os.path.join('.','output.txt'),'w+')
        fid.write('Liver,disease\n')
        for i in range(len(livers)):
            fid.write('{},{}\n'.format(livers[i],diseases[i]))
        fid.close()
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    # return_generators(False, path_extension='Single_Images3D_None', cube_size=(8, 32, 100, 100), return_test=False)
    pass
