"""
visualize index_i outputs from generate_test_output_for_input.py
"""

import numpy as np
from matplotlib import pyplot as plt

max_slice = 189
img_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/img.npy"
label_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/label.npy"
img = np.load(img_path) #(1, 224, 192, 1)
label = np.load(label_path)
for i in np.arange(max_slice):
  index_path = "".join(["/scratch/hasm/Data/Lesion/X-net_Test/X-Net/index_", str(i), ".npy"])
  index = np.load(index_path)
  total = img[0,:,:,]