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

data_file_path = "/scratch/hasm/Data/Lesion/ATLAS_R1.1/train.h5"
input_shape = (224, 192, 1)
num_patients = 229
parent_dir = os.path.join("/scratch/hasm", "Data", "Lesion")
xnet_dir = os.path.join(parent_dir, "X-net_Test",
                        "X-Net_20210401_CompleteDataSet_3Folds")
pretrained_weights_file = os.path.join(
    xnet_dir, "fold_2", "trained_final_weights.h5")
data_file_path = os.path.join(parent_dir, "ATLAS_R1.1", "train.h5")
output_dir = os.path.join(xnet_dir, "output_visualization")

if not (os.path.isfile(pretrained_weights_file)):
    print("".join(["ERROR: pretrained weightsfile (", str(
        pretrained_weights_file), ") does not exist."]), flush=True)
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
print("".join(["pretrained_weights_file: (", str(pretrained_weights_file), ")"]))

model = create_xception_unet_n(
    input_shape=input_shape, pretrained_weights_file=pretrained_weights_file)

print("Generated Model")

# val_patient_indexes = np.array([1])
for patient_index in np.arange(num_patients):
    val_patient_indexes = np.array([patient_index])
    print("".join(["val_patient_indexes: ", str(
        val_patient_indexes)]), flush=True)

    f = create_val_date_generator(
        patient_indexes=val_patient_indexes, h5_file_path=data_file_path)
    # img = nib.load(sample_input).get_fdata()
    # print("img shape")
    # print(np.shape(img))#(197, 233, 189)
    # sample_img = img[:,:,90]
    print("create_val_date_generator finished", flush=True)
    num_slices_val = len(val_patient_indexes) * 189
    print("".join(["num_slices_val: ", str(num_slices_val)]), flush=True)
    for slice_index in range(num_slices_val):
        img, label = f.__next__()
        # print(np.shape(img))
        # print(np.shape(label))
        np.save(os.path.join(output_dir, "".join(["patient_", str(
            patient_index), "_seg_", str(slice_index)]), model.predict(img)))
        np.save(os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_img_", str(slice_index)]), img))
        np.save(os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_gt_", str(slice_index)]), label))
