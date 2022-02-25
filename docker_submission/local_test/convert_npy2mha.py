import itk
import numpy as np

arr = np.load("/home/buzzwoll/CoNIC/local_test/npy/images.npy")
dump_itk = itk.image_from_array(arr[:20])
itk.imwrite(dump_itk, "/home/buzzwoll/CoNIC/local_test/mha/images_small.mha")

# dump_np = itk.imread("/mnt/c/users/Elias/Desktop/conic/CoNIC/local_test/mha/images.mha")
# dump_np = np.array(dump_np)

# assert np.sum(np.abs(dump_np-arr))==0