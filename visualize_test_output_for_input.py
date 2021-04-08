"""
visualize gt_i outputs from generate_test_output_for_input.py
"""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib

#assumes you are on CHPC or CHPC 2021 (Linux HPC Cluster); change detailed directories accordingly for each output
#username = os.getlogin()
username = "hasm" #override because CHPC returns root
lesion_dir = os.path.join("/scratch", str(username), "Data", "Lesion")
if ( username == "sungminha" ):
  xnet_dir = os.path.join(lesion_dir, "X-net" )
elif ( username == "hasm" ):
  xnet_dir = os.path.join(lesion_dir, "X-net_Test" )
else:
  print("".join(["ERROR: Unknown username (", str(username), "). Exiting"]))
  sys.exit()

output_dir = os.path.join(xnet_dir, "X-Net_20210401_CompleteDataSet_3Folds", "sample_test" )

if not (os.path.isdir(output_dir)):
    print("".join(["ERROR: output directory (", str(output_dir), ") does not exist."]), flush=True)
    sys.exit()

#reference - to get header from
reference_path = os.path.join(lesion_dir, "ATLAS_R1.1","Only_Data","Site1","031768","t01","031768_t1w_deface_stx.nii.gz")
if not (os.path.isfile(reference_path)):
    print("".join(["ERROR: reference (", str(reference_path), ") does not exist."]), flush=True)
    sys.exit()

dim_x = 224
dim_y = 192
dim_z = 189 #224x192x189 is full image; this may change with input data
# img_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/img.npy"
# label_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/label.npy"
# img = np.load(img_path) #(1, 224, 192, 1)
# label = np.load(label_path)

### start generating output images and output nii.gz

gt_matrix = np.zeros(shape = (dim_x, dim_y, dim_z))
img_matrix = np.zeros(shape = (dim_x, dim_y, dim_z))
label_matrix = np.zeros(shape = (dim_x, dim_y, dim_z))

for i in np.arange(dim_z):
  gt_path = os.path.join(output_dir, "".join(["index_", str(i), ".npy"]))
  img_path = os.path.join(output_dir, "".join(["img_", str(i), ".npy"]))
  label_path = os.path.join(output_dir, "".join(["label_", str(i), ".npy"]))

  if not (os.path.isfile(gt_path)):
    print("".join(["ERROR: gt file (", str(gt_path), ") does not exist."]), flush=True)
    sys.exit()
  if not (os.path.isfile(img_path)):
    print("".join(["ERROR: img file (", str(img_path), ") does not exist."]), flush=True)
    sys.exit()
  if not (os.path.isfile(label_path)):
    print("".join(["ERROR: label file (", str(label_path), ") does not exist."]), flush=True)
    sys.exit()

  img = np.load(img_path)
  label = np.load(label_path)
  gt = np.load(gt_path)
  del img_path
  del label_path
  del gt_path

  gt_matrix[:,:,i] = np.reshape(gt, newshape=(224,192))
  img_matrix[:,:,i] = np.reshape(img, newshape=(224,192))
  label_matrix[:,:,i] = np.reshape(label, newshape=(224,192))
  
  img = np.reshape(img, newshape=(224, 192))
  label = np.reshape(label, newshape=(224, 192))
  gt = np.reshape(gt, newshape=(224, 192))
  plt.imshow(img, cmap='gray') # I would add interpolation='none' #Image
  plt.imshow(label, cmap='Blues', alpha=0.5) # interpolation='none' #GT
  plt.imshow(gt, cmap='Reds', alpha=0.5) # interpolation='none' #Segmentation

  #plt.show()
  plt.savefig(os.path.join( output_dir, "".join(["gt_", str(i), ".png"]) ) )
  plt.close()
  del img
  del label
  del gt

#save img, gt, and label
reference = nib.load(reference_path)
empty_header = nib.Nifti1Header()

img_nii = nib.Nifti1Image(img_matrix, reference.affine, empty_header)
nib.save(img_nii, os.path.join( output_dir, 'img.nii.gz'))

label_nii = nib.Nifti1Image(label_matrix, reference.affine, empty_header)
nib.save(label_nii, os.path.join( output_dir, 'label.nii.gz'))

gt_nii = nib.Nifti1Image(gt_matrix, reference.affine, empty_header)
nib.save(gt_nii, os.path.join( output_dir, 'gt.nii.gz'))
