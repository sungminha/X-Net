"""
testing an input
"""
import keras
import keras.backend as K

import numpy as np

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard

from model import create_xception_unet_n
import nibabel as nib

data_file_path = "/scratch/hasm/Data/Lesion/ATLAS_R1.1/train.h5"
pretrained_weights_file = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/x_net_final_score.csv"
input_shape = (224, 192, 1)
sample_input = "/scratch/hasm/Data/Lesion/ATLAS_R1.1/Only_Data/Site1/031768/t01/031768_t1w_deface_stx.nii.gz"

print("".join(["data_file_path: (", str(data_file_path), ")"]))
print("".join(["input_shape: (", str(input_shape), ")"]))
print("".join(["pretrained_weights_file: (", str(pretrained_weights_file), ")"]))

model = create_xception_unet_n(input_shape=input_shape, pretrained_weights_file=pretrained_weights_file)

print("Generated Model")

img = nib.load(sample_input).get_fdata()

print(np.shape(img))
model.predict(img)
