import matplotlib.pyplot as plt
from src.metrics import *
import numpy as np
import shutil
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt

def plot_instances(X):
    image = Image.fromarray(X, mode='L')
    sh = X.shape
    image2 = Image.new(mode='RGB', size=sh)
    draw = image.load()
    X = X.astype(np.int32)
    for w in range(sh[1]):
        for h in range(sh[0]):
            image2.putpixel((h, w), tuple(clrs2[X[w,h]]))
    plt.xticks([])
    plt.yticks([])
    return np.asarray(image2)
clrs2 = [[0, 0, 0], [121, 186, 179], [173, 206, 184], [124, 183, 103], [58, 231, 42], [163, 200, 250], [50, 91, 30], [148, 149, 79], [19, 124, 78], [76, 70, 104], [46, 93, 122], [16, 198, 197], [210, 27, 187], [117, 113, 168], [200, 173, 237], [198, 145, 234], [243, 206, 201], [32, 74, 205], [33, 82, 156], [235, 19, 28], [228, 179, 135], [157, 39, 30], [122, 158, 156], [10, 57, 210], [36, 82, 49], [97, 105, 2], [154, 25, 251], [155, 53, 107], [17, 17, 170], [219, 87, 227], [247, 71, 221], [154, 194, 102], [216, 37, 83], [121, 66, 139], [9, 139, 190], [47, 99, 151], [238, 75, 214], [243, 160, 109], [94, 162, 5], [134, 0, 255], [137, 66, 227], [126, 238, 3], [185, 208, 44], [177, 95, 82], [18, 98, 153], [18, 192, 246], [79, 40, 179], [249, 84, 249], [21, 191, 252], [130, 214, 26], [133, 133, 29], [184, 253, 175], [237, 212, 101], [39, 37, 98], [150, 127, 237], [246, 235, 213], [38, 52, 213], [179, 113, 116], [79, 22, 30], [98, 239, 167], [135, 186, 225], [60, 196, 210], [34, 32, 3], [42, 128, 248], [202, 20, 87], [119, 15, 79], [230, 106, 30], [105, 136, 99], [178, 153, 66], [190, 83, 99], [116, 238, 155], [114, 110, 134], [252, 238, 8], [20, 86, 71], [159, 105, 35], [91, 210, 117], [63, 221, 46], [192, 172, 150], [235, 30, 2], [249, 254, 10], [24, 239, 127], [143, 126, 58], [161, 132, 207], [104, 43, 183], [167, 47, 219], [82, 242, 18], [154, 143, 45], [71, 73, 89], [173, 217, 251], [232, 170, 253], [101, 179, 169], [196, 180, 234], [199, 217, 102], [123, 136, 194], [170, 239, 58], [102, 52, 219], [74, 218, 201], [67, 41, 244], [69, 224, 20], [17, 9, 89], [150, 2, 226], [250, 217, 10], [176, 129, 247], [178, 95, 193], [172, 37, 115], [81, 71, 0], [42, 76, 140], [250, 112, 34], [192, 246, 139], [193, 127, 232], [218, 133, 30], [150, 115, 25], [220, 181, 154], [8, 219, 65], [224, 129, 28], [134, 68, 73], [221, 118, 68], [45, 69, 176], [40, 168, 139], [178, 132, 5], [232, 157, 48], [28, 198, 218], [103, 93, 15], [60, 76, 16], [213, 255, 60], [38, 146, 58], [12, 18, 148], [0, 116, 81], [185, 205, 51], [188, 32, 127], [150, 203, 214], [254, 180, 237], [97, 114, 120], [77, 19, 101], [47, 106, 161], [62, 41, 86], [54, 96, 144], [37, 79, 28], [139, 209, 172], [106, 47, 128], [101, 123, 64], [115, 146, 95], [45, 25, 223], [0, 127, 30], [184, 235, 118], [80, 48, 71], [255, 106, 233], [14, 18, 87], [118, 131, 71], [96, 109, 101], [164, 60, 82], [192, 41, 18], [255, 20, 213], [204, 67, 221], [104, 59, 150], [218, 45, 117], [12, 85, 188], [244, 11, 15], [28, 185, 183], [185, 131, 39], [127, 18, 145], [125, 195, 241], [70, 181, 186], [211, 246, 55], [161, 4, 68], [77, 226, 213], [90, 122, 219], [223, 146, 225], [171, 177, 59], [213, 201, 229], [126, 26, 204], [93, 151, 83], [247, 58, 48], [192, 129, 132], [103, 97, 126], [115, 20, 41], [191, 234, 171], [23, 63, 41], [98, 151, 251], [137, 160, 162], [6, 236, 115], [156, 160, 181], [3, 84, 106], [117, 232, 21], [93, 223, 58], [55, 143, 55], [0, 79, 42], [134, 78, 143], [194, 232, 100], [51, 234, 175], [79, 26, 108], [240, 205, 230], [233, 56, 201], [134, 15, 152], [143, 181, 174], [48, 204, 118], [81, 62, 237], [187, 127, 28], [170, 199, 85], [21, 217, 20], [247, 4, 63], [214, 120, 138], [240, 58, 128], [42, 174, 83], [161, 167, 151], [141, 241, 181], [205, 148, 248], [139, 23, 215], [210, 213, 84], [88, 220, 93], [101, 14, 161], [130, 231, 30], [35, 210, 45], [193, 130, 210], [74, 183, 86], [123, 149, 172], [70, 176, 221], [108, 116, 73], [245, 242, 172], [113, 165, 253], [158, 179, 29], [190, 74, 58], [111, 215, 126], [64, 125, 226], [161, 18, 72], [194, 174, 136], [211, 66, 100], [227, 126, 95], [111, 136, 121], [56, 184, 153], [187, 24, 50], [158, 64, 62], [96, 150, 187], [221, 134, 159], [91, 108, 104], [77, 181, 198], [123, 56, 10], [241, 73, 144], [219, 108, 99], [99, 47, 167], [153, 3, 195], [71, 9, 131], [28, 13, 173], [222, 114, 159], [183, 74, 62], [184, 217, 120], [179, 69, 108], [173, 4, 107], [246, 203, 42], [153, 191, 246], [220, 102, 102], [246, 140, 54], [223, 49, 0], [61, 10, 107], [221, 226, 255], [147, 106, 187], [180, 172, 137], [86, 16, 169], [61, 137, 127], [169, 138, 98], [70, 239, 213], [162, 156, 151], [11, 245, 98], [80, 175, 56], [151, 201, 220], [42, 203, 39], [246, 34, 161], [209, 43, 47], [185, 232, 214], [8, 161, 204], [47, 85, 141], [110, 110, 181], [90, 114, 215], [245, 165, 206], [250, 139, 245], [83, 92, 139], [1, 146, 248], [233, 234, 171], [114, 98, 114], [28, 50, 164], [197, 33, 245], [57, 107, 244], [19, 217, 213], [207, 112, 62], [151, 13, 168], [155, 194, 37], [23, 238, 7], [148, 121, 251], [240, 22, 89], [34, 66, 151], [137, 219, 252], [92, 107, 99], [68, 54, 129], [226, 225, 41], [102, 162, 238], [55, 242, 41], [220, 29, 89], [79, 124, 113], [33, 121, 0], [54, 87, 91], [7, 29, 147], [242, 217, 59], [163, 133, 234], [238, 6, 5], [226, 72, 138], [251, 225, 170], [216, 90, 131], [82, 111, 231], [220, 103, 101], [74, 236, 18], [228, 254, 87], [155, 200, 200], [177, 105, 143], [131, 74, 88], [128, 23, 210], [87, 102, 127], [210, 191, 186], [236, 236, 170], [182, 227, 231], [225, 255, 226], [254, 17, 177], [2, 12, 73], [159, 128, 4], [149, 169, 101], [155, 199, 101], [16, 26, 179], [50, 182, 187], [59, 122, 143], [40, 172, 181], [112, 87, 239], [33, 89, 160], [104, 221, 115], [167, 45, 13], [36, 122, 131], [83, 77, 44], [253, 253, 89], [18, 151, 127], [13, 123, 227], [221, 191, 154], [135, 120, 171], [16, 166, 226], [216, 199, 121], [161, 52, 51], [138, 242, 238], [71, 240, 60], [42, 78, 109], [139, 189, 53], [88, 64, 119], [89, 48, 231], [100, 230, 176], [206, 60, 17], [20, 34, 127], [86, 144, 157], [110, 33, 63], [3, 189, 228], [236, 239, 74], [193, 11, 168], [55, 111, 66], [250, 44, 55], [105, 133, 80], [78, 238, 78], [16, 168, 218], [242, 129, 228], [90, 161, 165], [245, 97, 251], [177, 63, 90], [79, 225, 218], [173, 174, 175], [86, 23, 250], [191, 97, 17], [234, 175, 141], [218, 147, 237], [15, 101, 92], [181, 71, 4], [124, 153, 111], [254, 183, 105], [182, 74, 71], [97, 42, 173], [177, 35, 95], [147, 215, 89], [131, 180, 111], [9, 177, 164], [159, 43, 112], [96, 40, 93], [65, 190, 119], [54, 188, 254], [84, 93, 223], [59, 50, 56], [130, 39, 115], [40, 215, 185], [194, 28, 206], [18, 224, 97], [184, 244, 41], [26, 69, 207], [115, 252, 61], [111, 177, 225], [96, 39, 25], [140, 210, 15], [47, 9, 193], [129, 187, 232], [89, 40, 218], [52, 165, 80], [142, 246, 232], [0, 219, 30], [122, 67, 60], [164, 249, 229], [222, 107, 252], [210, 46, 173], [189, 32, 252], [56, 185, 44], [141, 87, 213], [114, 185, 115], [109, 67, 103], [138, 32, 147], [82, 132, 172], [65, 204, 228], [21, 124, 163], [169, 248, 12], [253, 132, 157], [223, 207, 92], [122, 103, 114], [33, 103, 85], [238, 34, 75], [12, 7, 114], [78, 114, 183], [106, 48, 189], [25, 177, 144], [68, 104, 41], [114, 204, 201], [28, 31, 33], [63, 247, 111], [253, 18, 17], [61, 230, 13], [153, 233, 73], [191, 206, 216], [133, 129, 116], [53, 247, 205], [250, 139, 21], [170, 167, 110], [143, 62, 223], [237, 26, 254], [205, 110, 62], [230, 140, 205], [75, 101, 47], [149, 89, 184], [249, 10, 146], [221, 223, 220], [69, 48, 230], [125, 20, 69], [77, 226, 4], [240, 111, 5], [0, 16, 215], [244, 39, 232], [212, 103, 197], [87, 37, 251], [100, 165, 203], [160, 7, 14], [240, 6, 183], [160, 120, 221], [242, 155, 141], [100, 107, 126], [166, 180, 145], [15, 23, 121], [240, 207, 47], [1, 188, 235], [54, 54, 29], [217, 87, 182], [130, 209, 5], [83, 51, 234], [179, 22, 197], [99, 139, 254], [29, 168, 179], [182, 212, 28], [218, 192, 237], [182, 101, 50], [170, 132, 125], [74, 240, 87], [90, 86, 174], [18, 43, 63], [189, 51, 244], [229, 215, 223], [149, 40, 194], [185, 72, 105], [45, 124, 10], [19, 170, 20], [15, 41, 201], [87, 144, 7], [106, 205, 108], [40, 118, 86], [114, 52, 215], [102, 186, 182], [115, 87, 29], [117, 111, 48], [77, 70, 214], [83, 182, 194], [73, 96, 27], [174, 231, 36], [95, 63, 201], [62, 39, 106], [177, 188, 140], [17, 19, 208], [245, 16, 86], [171, 52, 37], [97, 128, 242], [113, 185, 93], [83, 205, 252], [159, 23, 108], [9, 48, 129], [59, 115, 99], [241, 150, 69], [125, 75, 241], [233, 66, 165], [139, 241, 73], [228, 0, 121], [157, 195, 160], [196, 151, 152], [182, 119, 35], [38, 207, 21], [60, 226, 19], [147, 99, 169], [213, 78, 194], [85, 56, 150], [132, 206, 49], [197, 138, 163], [141, 213, 225], [224, 191, 124], [136, 24, 130], [7, 111, 107], [145, 28, 52], [122, 188, 8], [93, 44, 199], [20, 227, 148], [158, 245, 193], [254, 78, 140], [8, 233, 124], [57, 250, 57], [224, 28, 95], [136, 190, 217], [240, 184, 131], [33, 250, 24], [29, 203, 145], [127, 161, 134], [119, 110, 152], [48, 197, 182], [137, 73, 1], [51, 168, 62], [68, 91, 22], [154, 11, 229], [116, 255, 195], [213, 29, 144], [26, 180, 76], [42, 13, 36], [109, 176, 121], [9, 114, 8], [77, 52, 97], [112, 129, 98], [221, 191, 190], [114, 41, 179], [138, 234, 178], [24, 15, 18], [180, 31, 76], [139, 153, 86], [16, 149, 176], [45, 113, 243], [237, 198, 64], [111, 206, 185], [28, 237, 25], [80, 147, 3], [182, 148, 1], [64, 50, 235], [163, 160, 1], [48, 209, 95], [87, 104, 121], [18, 119, 95], [133, 117, 91], [136, 168, 206], [168, 11, 99], [13, 46, 167], [63, 11, 51], [93, 207, 113], [169, 69, 240], [4, 66, 172], [6, 153, 133], [70, 132, 50], [82, 48, 21], [237, 241, 49], [123, 9, 74], [239, 110, 145], [187, 163, 11], [179, 50, 88], [188, 133, 12], [177, 106, 250], [81, 119, 196], [140, 108, 176], [238, 139, 71], [65, 245, 22], [254, 83, 180], [109, 142, 152], [127, 197, 26], [250, 207, 10], [149, 165, 171], [245, 230, 237], [20, 164, 37], [182, 50, 167], [138, 219, 151], [109, 63, 127], [90, 146, 129], [230, 120, 124], [226, 191, 110], [157, 105, 19], [62, 16, 42], [189, 54, 25], [114, 176, 243], [38, 74, 187], [105, 206, 116], [71, 37, 106], [119, 225, 41], [26, 38, 131], [213, 93, 251], [131, 164, 184], [68, 242, 133], [213, 105, 117], [202, 68, 240], [80, 42, 249], [116, 207, 188], [246, 162, 98], [207, 75, 100], [81, 183, 235], [143, 254, 149], [237, 145, 23], [57, 134, 145], [210, 82, 47], [22, 230, 216], [125, 127, 224], [32, 210, 254], [110, 140, 138], [87, 27, 186], [31, 148, 238], [31, 142, 142], [254, 252, 59], [121, 154, 105], [237, 52, 129], [239, 5, 143], [102, 51, 90], [71, 125, 219], [124, 91, 41], [8, 154, 56], [218, 56, 97], [189, 178, 160], [35, 178, 224], [252, 160, 215], [206, 31, 172], [87, 247, 25], [121, 80, 24], [170, 137, 94], [207, 30, 241], [82, 73, 151], [93, 42, 10], [76, 143, 245], [204, 7, 121], [61, 212, 6], [110, 105, 155], [176, 221, 8], [245, 57, 18], [71, 203, 142], [157, 138, 221], [97, 237, 135], [124, 116, 175], [189, 37, 238], [47, 50, 251], [195, 129, 204], [165, 250, 224], [234, 173, 98], [190, 233, 182], [65, 71, 212], [160, 177, 54], [169, 13, 160], [24, 41, 194], [9, 140, 157], [196, 106, 22], [111, 239, 83], [88, 162, 55], [233, 172, 255], [55, 166, 60], [233, 110, 66], [24, 152, 145], [243, 194, 55], [14, 214, 174], [53, 31, 167], [14, 16, 233], [32, 233, 166], [15, 47, 5], [244, 88, 192], [120, 187, 3], [193, 1, 20], [2, 88, 52], [127, 240, 254], [223, 214, 94], [48, 144, 121], [169, 214, 149], [94, 49, 0], [183, 147, 146], [56, 74, 73], [119, 144, 102], [24, 53, 28], [185, 73, 117], [194, 212, 226], [147, 67, 72], [109, 71, 224], [81, 52, 98], [194, 111, 32], [164, 39, 48], [24, 90, 214], [163, 185, 66], [3, 62, 57], [134, 144, 217], [10, 96, 173], [113, 241, 185], [191, 172, 148], [202, 51, 82], [107, 150, 3], [218, 192, 184], [224, 203, 52], [45, 178, 6], [202, 104, 7], [180, 122, 199], [106, 39, 230], [60, 224, 244], [189, 111, 45], [110, 225, 141], [202, 146, 62], [252, 61, 169], [51, 237, 205], [42, 198, 91], [165, 128, 190], [49, 95, 44], [123, 238, 209], [249, 168, 253], [96, 240, 22], [241, 77, 152], [158, 33, 214], [150, 77, 248], [109, 142, 235], [254, 239, 52], [167, 184, 137], [197, 63, 19], [168, 42, 248], [156, 14, 10], [90, 149, 247], [23, 5, 213], [152, 249, 90], [157, 105, 178], [56, 50, 208], [133, 220, 40], [238, 233, 206], [20, 231, 233], [198, 180, 236], [210, 94, 205], [73, 115, 37], [221, 255, 123], [5, 85, 152], [27, 210, 212], [13, 180, 4], [237, 53, 39], [109, 17, 223], [170, 117, 164], [193, 147, 161], [228, 54, 8], [159, 48, 182], [128, 215, 3], [161, 254, 188], [253, 49, 234], [144, 72, 113], [127, 160, 101], [248, 224, 6], [174, 242, 21], [180, 58, 236], [120, 80, 43], [48, 212, 44], [225, 26, 217], [57, 167, 232], [196, 15, 7], [87, 84, 78], [69, 154, 61], [30, 40, 125], [184, 176, 204], [189, 113, 66], [209, 39, 147], [20, 161, 125], [153, 8, 78], [30, 39, 238], [22, 100, 17], [176, 180, 127], [218, 47, 29], [136, 247, 232], [207, 125, 42], [64, 8, 253], [251, 210, 30], [4, 39, 94], [157, 47, 144], [43, 243, 112], [213, 80, 248], [192, 170, 104], [23, 31, 1], [169, 190, 241], [225, 169, 179], [34, 157, 163], [30, 145, 44], [97, 76, 201], [211, 253, 66], [123, 65, 214], [86, 129, 81], [63, 11, 92], [93, 169, 124], [173, 73, 54], [67, 206, 155], [64, 28, 108], [3, 227, 149],]

def get_pq(true, pred, match_iou=0.5, remap=True):
    """Get the panoptic quality result. 
    
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag 
    has no effect on the result.
    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5, 
            Munkres assignment (solving minimum weight matching in bipartite graphs) 
            is caculated to find the maximal amount of unique pairing. If 
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.  
        remap (bool): whether to ensure contiguous ordering of instances.
    
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
        
        paired_iou.sum(): sum of IoU within true positive predictions
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    if remap:
        pred = remap_label(pred)
        true = remap_label(true)

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask_lab = true == true_id
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
        t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
        t_mask_crop = t_mask_crop.astype("int")
        p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
        pred_true_overlap = p_mask_crop[t_mask_crop > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask_lab = pred == pred_id
            p_mask_lab = p_mask_lab.astype("int")

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
            p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        paired_iou.sum(),
        [paired_true, unpaired_pred, unpaired_true],
        [pred,true],
        pairwise_iou
    )

def get_multi_pq_info(true, pred, nr_classes=6, match_iou=0.5):
    """Get the statistical information needed to compute multi-class PQ.
    
    CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time, 
    rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
    when a nuclear category is not present in a particular image.
    
    Args:
        true (ndarray): HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map. 
        pred: HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map. 
        nr_classes (int): Number of classes considered in the dataset. 
        match_iou (float): IoU threshold for determining whether there is a detection.
    
    Returns:
        statistical info per class needed to compute PQ.
    
    """

    assert match_iou >= 0.0, "Cant' be negative"

    true_inst = true[..., 0]
    pred_inst = pred[..., 0]
    ###
    true_class = true[..., 1]
    pred_class = pred[..., 1]

    pq = []
    for idx in range(nr_classes):
        pred_class_tmp = pred_class == idx + 1
        pred_inst_oneclass = pred_inst * pred_class_tmp
        pred_inst_oneclass = remap_label(pred_inst_oneclass)
        ##
        true_class_tmp = true_class == idx + 1
        true_inst_oneclass = true_inst * true_class_tmp
        true_inst_oneclass = remap_label(true_inst_oneclass)

        [DQ, SQ, PQ],[tp,fp,fn],paired_iou,[paired_true, unpaired_pred, unpaired_true],[_,_],pairwise_iou = get_pq(true_inst_oneclass, pred_inst_oneclass, remap=False)

        pq_oneclass_stats = [
            PQ,
            [paired_true, unpaired_pred, unpaired_true], # indizes
            [true_inst_oneclass, pred_inst_oneclass]
        ]
        pq.append(pq_oneclass_stats)
    return pq, 


from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import generate_binary_structure

def inst_to_3c_cp(gt_labels, sigma = 2):
    '''
    Generate 3class map and center of masses for each instance
    '''
    gt_labels = np.squeeze(gt_labels)
    gt_threeclass = np.zeros(gt_labels.shape, dtype=np.uint8)
    gt_centers = np.zeros(gt_labels.shape, dtype=np.float32)
    struct = generate_binary_structure(2,2) # TODO can replace this with a np.full([3,3], True)
    crds = np.array(np.meshgrid(np.arange(gt_labels.shape[1]), np.arange(gt_labels.shape[0]))).transpose(1,2,0)
    gt_coords = []
    for inst in np.unique(gt_labels):
        if inst == 0:
            continue
        lab = gt_labels == inst
        tmp = crds[lab]
        y,x = np.median(tmp, axis=0).astype(np.int32) #Take median instead of mean of coordinates
        if gt_labels[x,y] == 0:
            y,x = get_closest_to_median(np.array([y,x]), tmp.T)
        # TODO we never check if this point is actually in the cell, however I would argue that 
        # x,y = center_of_mass(lab) 
        eroded_lab = binary_erosion(lab, iterations=1, structure=struct, border_value=1)
        boundary = np.logical_xor(lab, eroded_lab)
        gt_threeclass[boundary] = 2
        gt_threeclass[eroded_lab] = 1
        gt_centers[int(x),int(y)]  = 1.
        gt_coords.append([inst,x,y])
    # gt_centers = gaussian_filter(gt_centers, sigma, mode='constant')
    # max_value = np.max(gt_centers)
    # if max_value >0:
    #     gt_centers /= max_value # rescale to [0,1]
    # gt_centers = gt_centers.astype(np.float32)
    return gt_threeclass[np.newaxis,], gt_centers[np.newaxis,], np.array(gt_coords)[np.newaxis,]

def get_closest_to_median(crd, all):
    sq_dist = (all**2).sum(0) + crd.dot(crd) - 2*crd.dot(all)
    return all[:,int(sq_dist.argmin())]


title_size = 16
for gt, pred, [raw, _] in tqdm(zip(gt_list, pred_list, validation_dataset)):
    fig, ax = plt.subplots(2,8, figsize=(44,12))
    ax[0,0].imshow(raw)
    ax[1,0].imshow(plot_instances(gt[...,0]), interpolation='nearest')
    ax[0,1].imshow(plot_instances(gt[...,1]), interpolation='nearest')
    ax[1,1].imshow(plot_instances(pred[...,1]))
    ax[0,0].title.set_text('raw')
    ax[1,0].title.set_text('gt instances')
    ax[0,1].title.set_text('gt ct')
    ax[1,1].title.set_text('pred_ct')
    ax[0,0].title.set_size(title_size)
    ax[0,1].title.set_size(title_size)
    ax[1,0].title.set_size(title_size)
    ax[1,1].title.set_size(title_size)
    metr = get_multi_pq_info(true = gt, pred = pred)
    x = 2
    PQ_l = []
    for [PQ, [tp_idx, fp_idx, fn_idx], [true_inst_oneclass, pred_inst_oneclass]] in metr[0]:    
        gt_tmp = true_inst_oneclass
        pred_tmp = pred_inst_oneclass
        tmp = np.zeros([256,256,3])
        for idx in tp_idx:
            tmp[gt_tmp==idx,1]=1 # green tp
        for idx in fp_idx:
            tmp[pred_tmp==idx,0]=1 # red fp
        for idx in fn_idx:
            tmp[gt_tmp==idx,2]=1  # blue fn
        ax[1,x].imshow(tmp)
        ax[0,x].imshow(np.stack([true_inst_oneclass>0, np.zeros_like(true_inst_oneclass), pred_inst_oneclass>0],axis=-1).astype(np.float32))
        ax[0,x].title.set_text('GT red, Pred blue, class '+str(x-1))
        ax[1,x].title.set_text('TP:green, FP: red, FN: blue, PQ: '+str(PQ)[:5])
        ax[0,x].title.set_size(title_size)
        ax[1,x].title.set_size(title_size)
        x += 1
        PQ_l.append(PQ)
    PQ = np.mean([PQ_l[i-1] for i in list(np.unique(gt[...,1]).astype(np.uint8))])
    ax[0,1].title.set_text('gt ct, mPQ: '+str(PQ)[:5])
    ax[0,1].title.set_size(title_size)
    plt.tight_layout()
    plt.savefig('samples/mPQ_'+str(PQ)[:6]+'.png', facecolor='white')
    plt.show()