"""
testing an input

need to run export HDF5_USE_FILE_LOCKING='FALSE' first in bash
"""
import keras
import keras.backend as K

import os
import sys
import numpy as np

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard

from model import create_xception_unet_n
import nibabel as nib
from data import create_train_date_generator, create_val_date_generator

# data_file_path = "/scratch/hasm/Data/Lesion/ATLAS_R1.1/train.h5"
data_file_path = "/home/hasm/comp_space/Data/Lesion/ATLAS_R1.1/Sample_Visualization/train.h5"
input_shape = (224, 192, 1)
# num_patients = 229
num_patients = 2
num_slices = 189
parent_dir = os.path.join("/scratch/hasm", "Data", "Lesion")
xnet_dir = os.path.join(parent_dir, "X-net_Test",
                        "X-Net_20210413_SampleVisualization")
attention_dir = os.path.join(parent_dir, "AttentionUnet_Test")
#for xnet
xnet_pretrained_weights_file = os.path.join(
    parent_dir, "X-net_Test", "X-Net_20210401_CompleteDataSet_3Folds", "fold_2", "trained_final_weights.h5")
#for attention unet
attention_pretrained_weights_file = os.path.join(
    attention_dir, "trained_models.h5")

# data_file_path = os.path.join(parent_dir, "ATLAS_R1.1", "train.h5")
data_file_path = os.path.join(parent_dir, "ATLAS_R1.1", "Sample_Visualization", "train.h5")

#part 1: xnet outputs
output_dir = os.path.join(xnet_dir, "output_visualization")

if not (os.path.isfile(xnet_pretrained_weights_file)):
    print("".join(["ERROR: Xnet pretrained weights file (", str(
        xnet_pretrained_weights_file), ") does not exist."]), flush=True)
    sys.exit()
if not (os.path.isfile(data_file_path)):
    print("".join(
        ["ERROR: data file (", str(data_file_path), ") does not exist."]), flush=True)
    sys.exit()
if not (os.path.isdir(output_dir)):
    print("".join(["ERROR: output directory (", str(
        output_dir), ") does not exist."]), flush=True)
    sys.exit()

# sample_input = "/scratch/hasm/Data/Lesion/ATLAS_R1.1/Only_Data/Site1/031768/t01/031768_t1w_deface_stx.nii.gz"

print("".join(["data_file_path: (", str(data_file_path), ")"]))
print("".join(["input_shape: (", str(input_shape), ")"]))
print("".join(["xnet_pretrained_weights_file: (", str(xnet_pretrained_weights_file), ")"]))

model = create_xception_unet_n(
    input_shape=input_shape, pretrained_weights_file=xnet_pretrained_weights_file)

print("Generated Model")

# val_patient_indexes = np.array([1])
for patient_index in np.arange(num_patients):
    val_patient_indexes = np.array([patient_index])
    num_slices_val = len(val_patient_indexes) * num_slices

    print("".join(["val_patient_indexes: ", str(
        val_patient_indexes)]), flush=True)

    output_path_seg_final = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_seg_", str(num_slices_val-1)]))
    output_path_img_final = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_img_", str(num_slices_val-1)]))
    output_path_gt_final = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_gt_", str(num_slices_val-1)]))
    if (os.path.isfile("".join([output_path_gt_final, ".npy"]))):
        print("".join(
            ["output_path_gt_final (", str("".join([output_path_gt_final, ".npy"])), ") already exists. Skipping patient ", str(patient_index)]), flush=True)
        continue

    f = create_val_date_generator(
        patient_indexes=val_patient_indexes, h5_file_path=data_file_path)
    # img = nib.load(sample_input).get_fdata()
    # print("img shape")
    # print(np.shape(img))#(197, 233, 189)
    # sample_img = img[:,:,90]
    print("create_val_date_generator finished", flush=True)
    print("".join(["num_slices_val: ", str(num_slices_val)]), flush=True)
    for slice_index in np.arange(num_slices_val):
        print("".join(["patient_index | slice_index: ", str(
            patient_index), "\t|\t", str(slice_index)]), flush=True)
        output_path_seg = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_seg_", str(slice_index)]))
        output_path_img = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_img_", str(slice_index)]))
        output_path_gt = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_gt_", str(slice_index)]))
        if (os.path.isfile(output_path_gt)):
            print("".join(
                ["output_path_gt (", str("".join([output_path_gt, ".npy"])), ") already exists. Skipping patient ", str(patient_index)]), flush=True)
            continue
        img, label = f.__next__()
        # print(np.shape(img))
        # print(np.shape(label))
        # print("patient_index:")
        # print(patient_index)
        # print("num_slices:")
        # print(num_slices)
        np.save(output_path_seg, model.predict(img))
        np.save(output_path_img, img)
        np.save(output_path_gt, label)


#part 1: attention outputs
output_dir = os.path.join(attention_dir, "output_visualization")
print("".join(["output_dir: (", str(output_dir),")"]))

if not (os.path.isfile(attention_pretrained_weights_file)):
    print("".join(["ERROR: Attention pretrained weights file (", str(
        attention_pretrained_weights_file), ") does not exist."]), flush=True)
    sys.exit()
if not (os.path.isfile(data_file_path)):
    print("".join(
        ["ERROR: data file (", str(data_file_path), ") does not exist."]), flush=True)
    sys.exit()
if not (os.path.isdir(output_dir)):
    print("".join(["ERROR: output directory (", str(
        output_dir), ") does not exist."]), flush=True)
    sys.exit()

# sample_input = "/scratch/hasm/Data/Lesion/ATLAS_R1.1/Only_Data/Site1/031768/t01/031768_t1w_deface_stx.nii.gz"

print("".join(["data_file_path: (", str(data_file_path), ")"]))
print("".join(["input_shape: (", str(input_shape), ")"]))
print("".join(["attention_pretrained_weights_file: (", str(attention_pretrained_weights_file), ")"]))

model = create_xception_unet_n(
    input_shape=input_shape, pretrained_weights_file=attention_pretrained_weights_file)

print("Generated Model")

# val_patient_indexes = np.array([1])
for patient_index in np.arange(num_patients):
    val_patient_indexes = np.array([patient_index])
    num_slices_val = len(val_patient_indexes) * num_slices

    print("".join(["val_patient_indexes: ", str(
        val_patient_indexes)]), flush=True)

    output_path_seg_final = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_seg_", str(num_slices_val-1)]))
    output_path_img_final = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_img_", str(num_slices_val-1)]))
    output_path_gt_final = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_gt_", str(num_slices_val-1)]))
    if (os.path.isfile("".join([output_path_gt_final, ".npy"]))):
        print("".join(
            ["output_path_gt_final (", str("".join([output_path_gt_final, ".npy"])), ") already exists. Skipping patient ", str(patient_index)]), flush=True)
        continue

    f = create_val_date_generator(
        patient_indexes=val_patient_indexes, h5_file_path=data_file_path)
    # img = nib.load(sample_input).get_fdata()
    # print("img shape")
    # print(np.shape(img))#(197, 233, 189)
    # sample_img = img[:,:,90]
    print("create_val_date_generator finished", flush=True)
    print("".join(["num_slices_val: ", str(num_slices_val)]), flush=True)
    for slice_index in np.arange(num_slices_val):
        print("".join(["patient_index | slice_index: ", str(
            patient_index), "\t|\t", str(slice_index)]), flush=True)
        output_path_seg = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_seg_", str(slice_index)]))
        output_path_img = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_img_", str(slice_index)]))
        output_path_gt = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_gt_", str(slice_index)]))
        if (os.path.isfile(output_path_gt)):
            print("".join(
                ["output_path_gt (", str("".join([output_path_gt, ".npy"])), ") already exists. Skipping patient ", str(patient_index)]), flush=True)
            continue
        img, label = f.__next__()
        # print(np.shape(img))
        # print(np.shape(label))
        # print("patient_index:")
        # print(patient_index)
        # print("num_slices:")
        # print(num_slices)
        np.save(output_path_seg, model.predict(img))
        np.save(output_path_img, img)
        np.save(output_path_gt, label)

