import itk
import numpy as np


dump_np = np.array(itk.imread("/home/buzzwoll/CoNIC/local_test/output/pred_seg.mha"))
img = dump_np[0,:,:,0]
img1 = dump_np[0,:,:,1]

itk.imwrite(itk.image_from_array(img1.astype(np.uint8)), "/home/buzzwoll/CoNIC/local_test/output/test1.png")

print("test")