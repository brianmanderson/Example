__author__ = 'Brian M Anderson'
# Created on 9/2/2020
import sys, os

sys.path.append('..')
if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def main():
    from Return_Train_Validation_Generators_TF2 import return_paths
    base_path, morfeus_drive = return_paths()
    from Return_Train_Validation_Generators_TF2 import get_layers_dict_dense_less_decode, my_UNet, return_generators, \
        get_layers_dict_dense_HNet, return_model, plot_scroll_Image, os
    import numpy as np
    import SimpleITK as sitk
    add = '_16'
    model_name = 'DenseNetNewMultiBatch'
    cache_add = ''

    make_pred = False
    if make_pred:
        model_path = os.path.join(base_path, 'Keras', model_name, 'Models', 'Trial_ID_42', 'Model')
        layers_dict = get_layers_dict_dense_HNet(layers=2, filters=32, num_conv_blocks=2,
                                                 conv_lambda=2, max_conv_blocks=12)
        weights_path = os.path.join(base_path, 'Keras', model_name, 'Models', 'Trial_ID_42', 'cp-0031.h5')
        model = return_model(layers_dict=layers_dict, densenet=True, all_trainable=True, weights_path=weights_path)
        if not os.path.exists(model_path):
            model.save(model_path)
            return None
        # model = tf.keras.models.load_model(model_path)
        for is_test in [False]:
            if is_test:
                ext = 'Test'
            else:
                ext = 'Validation'
            base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=1, add=add,
                                                                                                cache_add=cache_add,
                                                                                                flip=True,
                                                                                                change_background=False,
                                                                                                threshold=True,
                                                                                                threshold_val=10,
                                                                                                path_lead='Records',
                                                                                                validation_name='',
                                                                                                is_test=is_test,
                                                                                                cache=False,
                                                                                                wanted_keys={'inputs': ['image', 'mask', 'image_path'], 'outputs': ['annotation']})
            generator = validation_generator.data_set.as_numpy_iterator()
            if not os.path.exists(os.path.join(base_path, 'Predictions_np', ext)):
                os.makedirs(os.path.join(base_path, 'Predictions_np', ext))
            for i in range(len(validation_generator)):
                print(i)
                x, y = next(generator)
                file_name = os.path.split(x[-1][0].decode())[-1]
                print(file_name)
                x = x[:-1]
                print(x[0].shape)
                step = 96
                shift = 120 // 2
                gap = 20 // 2
                if x[0].shape[1] > step:
                    pred = np.zeros(x[0][0].shape[:-1] + (2,))
                    start = 0
                    while start < x[0].shape[1]:
                        image_cube, mask_cube = x[0][:, start:start + step, ...], x[1][:, start:start + step, ...]
                        difference = image_cube.shape[1] % 32
                        if difference != 0:
                            image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                            mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                        pred_cube = model.predict([image_cube, mask_cube])
                        pred_cube = pred_cube[:, difference:, ...]
                        start_gap = gap
                        stop_gap = gap
                        if start == 0:
                            start_gap = 0
                        elif start + step >= x[0].shape[1]:
                            stop_gap = 0
                        if stop_gap != 0:
                            pred_cube = pred_cube[:, start_gap:-stop_gap, ...]
                        else:
                            pred_cube = pred_cube[:, start_gap:, ...]
                        pred[start + start_gap:start + start_gap + pred_cube.shape[1], ...] = pred_cube[0, ...]
                        start += shift
                else:
                    image_cube, mask_cube = x[0], x[1]
                    difference = image_cube.shape[1] % 32
                    if difference != 0:
                        image_cube = np.pad(image_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                        mask_cube = np.pad(mask_cube, [[0, 0], [difference, 0], [0, 0], [0, 0], [0, 0]])
                    pred_cube = model.predict([image_cube, mask_cube])
                    pred = pred_cube[:, difference:, ...]
                np.save(os.path.join(base_path, 'Predictions_np', ext, 'Prediction_{}.npy'.format(file_name)), pred)
                np.save(os.path.join(base_path, 'Predictions_np', ext, 'Image_{}.npy'.format(file_name)), np.squeeze(x[0]))
                np.save(os.path.join(base_path, 'Predictions_np', ext, 'Truth_{}.npy'.format(file_name)), np.squeeze(y[0]))

    create_nifti_files = False
    if create_nifti_files:
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        for ext in ['Validation']:
            input_path = r'H:\Liver_Disease_Ablation\{}'.format(ext)
            out_path = r'H:\Liver_Disease_Ablation\Predictions_HNet\{}'.format(ext)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            path = os.path.join(base_path, 'Predictions_np', ext)
            images = [i for i in os.listdir(path) if i.startswith('Image')]
            for file in images:
                iteration_number = file.split('LiTs_')[-1].split('.nii.')[0]
                if os.path.exists(os.path.join(out_path, '{}_Image.nii.gz'.format(iteration_number))):
                    continue
                print(file)
                x = np.load(os.path.join(path, file))
                pred = np.load(os.path.join(path, file.replace('Image', 'Prediction')))
                truth = np.load(os.path.join(path, file.replace('Image', 'Truth')))
                reader.SetFileName(os.path.join(input_path, 'Overall_Data_LiTs_{}.nii.gz'.format(iteration_number)))
                reader.Execute()
                image_handle = sitk.GetImageFromArray(x)
                pred_handle = sitk.GetImageFromArray(np.squeeze(pred[..., -1]))
                truth_handle = sitk.GetImageFromArray(truth.astype('int'))
                for handle in [image_handle, pred_handle, truth_handle]:
                    handle.SetOrigin(reader.GetOrigin())
                    handle.SetSpacing(reader.GetSpacing())
                    handle.SetDirection(reader.GetDirection())
                sitk.WriteImage(image_handle, os.path.join(out_path, '{}_Image.nii.gz'.format(iteration_number)))
                sitk.WriteImage(pred_handle, os.path.join(out_path, '{}_Prediction.nii.gz'.format(iteration_number)))
                sitk.WriteImage(truth_handle, os.path.join(out_path, '{}_Truth.nii.gz'.format(iteration_number)))


    evaluate_prediction = True
    if evaluate_prediction:
        from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart, np, cpu_count
        path = r'H:\Liver_Disease_Ablation\Predictions_HNet\Validation'
        create_metric_chart(path=path, out_path=os.path.join('.', 'Evaluate_Model', 'Threshold_Seed_Pickles_HNet'),
                            seed_range=np.arange(.05, .4, 0.05),
                            threshold_range=np.arange(0.05, 0.5, 0.05), re_write=False, thread_count=20)

    evaluate_test = False
    if evaluate_test:
        from Deep_Learning.Evaluate_Model.Evaluate_On_Data_TF2 import create_metric_chart, np
        path = r'H:\Liver_Disease_Ablation\Predictions_HNet\Test'
        create_metric_chart(path=path, out_path=os.path.join('.', 'Evaluate_Model', 'Test_HNet_Whole_Patient'),
                            seed_range=[0.65], write_final_prediction=True, single_disease=False,
                            threshold_range=[.25], re_write=False, thread_count=5)
        create_metric_chart(path=path, out_path=os.path.join('.', 'Evaluate_Model', 'Test_HNet'),
                            seed_range=[0.65], write_final_prediction=True, single_disease=True,
                            threshold_range=[.25], re_write=False, thread_count=5)

    write_sensitivity_specificity = False
    if write_sensitivity_specificity:
        from Deep_Learning.Evaluate_Model.Sensitivity_and_Specificity_Measures import write_sensitivity_specificity
        write_sensitivity_specificity(excel_path=os.path.join('.', 'Evaluate_Model', 'Sensitivity_and_FP.xlsx'))

    write_box_plots = False
    if write_box_plots:
        from Deep_Learning.Evaluate_Model.Make_Box_Plots import create_plot
        import pandas as pd
        import numpy as np
        out_path = os.path.join('.', 'Test_Output','Final_Prediction.xlsx')
        df = pd.read_excel(out_path, engine='openpyxl')
        volumes = df['volume'].values
        values = df['dice'].values
        values = values[volumes < 1]
        values = values[values < 1000]
        create_plot('Dice for less than 1 cc volumes',values=values,metric='Dice', out_path=os.path.join('.','Images'),
                    y_ticks=[0,.1,.2,.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        xxx = 1


if __name__ == '__main__':
    main()
