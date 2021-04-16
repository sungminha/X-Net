"""
visualize gt_i outputs from generate_test_output_for_input.py
"""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib

import argparse


# assumes you are on CHPC or CHPC 2021 (Linux HPC Cluster); change detailed directories accordingly for each output
#username = os.getlogin()
username = "hasm"  # override because CHPC returns root
lesion_dir = os.path.join("/scratch", str(username), "Data", "Lesion")
if (username == "sungminha"):
    xnet_dir = os.path.join(lesion_dir, "X-net")
elif (username == "hasm"):
    xnet_dir = os.path.join(lesion_dir, "X-net_Test")
else:
    print("".join(["ERROR: Unknown username (", str(username), "). Exiting"]))
    sys.exit()

#xnet
# output_dir = os.path.join(
    # xnet_dir, "X-Net_20210401_CompleteDataSet_3Folds", "output_visualization")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num-patients',
    default="229",
    type=int,
    dest='num_patients',
    help='number of subjects (int)'
)
parser.add_argument(
    '--output-dir',
    default=os.path.join(xnet_dir, "output_visualization"),
    type=str,
    dest='output_dir',
    help='output directory full path'
)
args = parser.parse_args()

# num_patients = 229
# num_patients = 3
num_patients = args.num_patients

# output dir
output_dir = args.output_dir

if not (os.path.isdir(output_dir)):
    print("".join(["ERROR: output directory (", str(
        output_dir), ") does not exist."]), flush=True)
    sys.exit()

# reference - to get header from
reference_path = os.path.join(lesion_dir, "ATLAS_R1.1", "Only_Data",
                              "Site1", "031768", "t01", "031768_t1w_deface_stx.nii.gz")
if not (os.path.isfile(reference_path)):
    print("".join(
        ["ERROR: reference (", str(reference_path), ") does not exist."]), flush=True)
    sys.exit()

dim_x = 224
dim_y = 192
dim_z = 189  # 224x192x189 is full image; this may change with input data
# img_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/img.npy"
# seg_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/seg.npy"
# gt_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/gt.npy"
# img = np.load(img_path) #(1, 224, 192, 1)

# start generating output images and output nii.gz

for patient_index in np.arange(num_patients):
    output_path_img = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_img.nii.gz"]))
    output_path_seg = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_seg.nii.gz"]))
    output_path_gt = os.path.join(output_dir, "".join(
        ["patient_", str(patient_index), "_gt.nii.gz"]))
    if (os.path.isfile(output_path_gt)):
        print("".join(
            ["output_path_gt (", str(output_path_gt), ") already exists. Skipping patient ", str(patient_index)]), flush=True)
        continue

    gt_matrix = np.zeros(shape=(dim_x, dim_y, dim_z))  # ground truth
    img_matrix = np.zeros(shape=(dim_x, dim_y, dim_z))  # image
    seg_matrix = np.zeros(shape=(dim_x, dim_y, dim_z))  # segmentation

    for i in np.arange(dim_z):
        seg_path = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_seg_", str(i), ".npy"]))
        img_path = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_img_", str(i), ".npy"]))
        gt_path = os.path.join(output_dir, "".join(
            ["patient_", str(patient_index), "_gt_", str(i), ".npy"]))

        if not (os.path.isfile(gt_path)):
            print("".join(["ERROR: ground truth file (", str(
                gt_path), ") does not exist."]), flush=True)
            sys.exit()
        if not (os.path.isfile(img_path)):
            print(
                "".join(["ERROR: img file (", str(img_path), ") does not exist."]), flush=True)
            sys.exit()
        if not (os.path.isfile(seg_path)):
            print("".join(["ERROR: segmentation file (", str(
                seg_path), ") does not exist."]), flush=True)
            sys.exit()

        img = np.load(img_path)
        seg = np.load(seg_path)
        gt = np.load(gt_path)
        del img_path
        del seg_path
        del gt_path

        gt_matrix[:, :, i] = np.reshape(gt, newshape=(224, 192))
        img_matrix[:, :, i] = np.reshape(img, newshape=(224, 192))
        seg_matrix[:, :, i] = np.reshape(seg, newshape=(224, 192))

        img = np.reshape(img, newshape=(224, 192))
        seg = np.reshape(seg, newshape=(224, 192))
        gt = np.reshape(gt, newshape=(224, 192))
        plt.imshow(img, cmap='gray')  # I would add interpolation='none' #Image
        plt.imshow(seg, cmap='Blues', alpha=0.5)  # interpolation='none' #GT
        # interpolation='none' #Segmentation
        plt.imshow(gt, cmap='Reds', alpha=0.5)

        # plt.show()
        plt.savefig(os.path.join(output_dir, "".join(["patient_", str(patient_index), "_slice_", str(i), ".png"])))
        plt.close()
        del img
        del seg
        del gt

    # save img, gt, and seg
    reference = nib.load(reference_path)
    empty_header = nib.Nifti1Header()

    img_nii = nib.Nifti1Image(img_matrix, reference.affine, empty_header)

    nib.save(img_nii, output_path_img)
    del img_matrix
    del img_nii

    seg_nii = nib.Nifti1Image(seg_matrix, reference.affine, empty_header)
    nib.save(seg_nii, output_path_seg)
    del seg_matrix
    del seg_nii

    gt_nii = nib.Nifti1Image(gt_matrix, reference.affine, empty_header)
    nib.save(gt_nii, output_path_gt)
    del gt_matrix
    del gt_nii

    del output_path_img
    del output_path_seg
    del output_path_gt
