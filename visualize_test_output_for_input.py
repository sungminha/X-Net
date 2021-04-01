"""
visualize index_i outputs from generate_test_output_for_input.py
"""

import numpy as np
from matplotlib import pyplot as plt

max_slice = 189
# img_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/img.npy"
# label_path = "/scratch/hasm/Data/Lesion/X-net_Test/X-Net/label.npy"
# img = np.load(img_path) #(1, 224, 192, 1)
# label = np.load(label_path)
for i in np.arange(max_slice):
  index_path = "".join(["/scratch/hasm/Data/Lesion/X-net_Test/X-Net/index_", str(i), ".npy"])
  img_path = "".join(["/scratch/hasm/Data/Lesion/X-net_Test/X-Net/img_", str(i), ".npy"])
  label_path = "".join(["/scratch/hasm/Data/Lesion/X-net_Test/X-Net/label_", str(i), ".npy"])
  img = np.load(img_path)
  label = np.load(label_path)
  index = np.load(index_path)
  
  img = np.reshape(img, newshape=(224, 192))
  label = np.reshape(label, newshape=(224, 192))
  index = np.reshape(index, newshape=(224, 192))
  plt.imshow(img, cmap='gray') # I would add interpolation='none'
  plt.imshow(label, cmap='Blues', alpha=0.5) # interpolation='none'
  plt.imshow(index, cmap='Reds', alpha=0.5) # interpolation='none'

  #plt.show()
  plt.savefig("".join(["index_", str(i), ".png"]))
  plt.close()
  del img
  del label
  del index
  del img_path
  del label_path
  del index_path