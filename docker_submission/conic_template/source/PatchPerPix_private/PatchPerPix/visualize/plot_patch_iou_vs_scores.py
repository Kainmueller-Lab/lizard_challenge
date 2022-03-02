# python plot_patch_iou_vs_scores.py <setup>/test/processed/700000  0_85

import glob
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import zarr
import neurolight.gunpowder as nl
import scipy.stats


def get_scores_iou(fn, th):

    iou_key = "volumes/{}/IOU".format(th)
    scores_key = "volumes/{}/scores".format(th)

    fl = zarr.open(fn, 'r')
    iou = np.array(fl[iou_key])
    scores = np.array(fl[scores_key])

    return list(scores.ravel()), list(iou.ravel())

if __name__ == "__main__":
    in_df = sys.argv[1]
    th = sys.argv[2]

    scores = []
    iou = []

    if not os.path.exists(in_df):
        raise RuntimeError("invalid input")
    if in_df.endswith("zarr"):
        fns = [in_df]
    else:
        fns = glob.glob(os.path.join(in_df, "*.zarr"))
    print(fns)
    for fn in fns:
        s, i = get_scores_iou(fn, th)
        scores.extend(s)
        iou.extend(i)

    print(len(scores), len(iou))

    scores = np.array(scores)
    tmp = np.copy(scores)
    tmp[tmp > 0] = tmp[tmp > 0] - 100
    print(np.min(tmp), np.max(tmp))
    iou = np.array(iou)
    scores_bin = scores > 0
    print(np.sum(scores_bin))
    iou_bin = iou > 0

    print(np.sum(scores[scores_bin > 0] == 100))
    scores[scores_bin > 0] = scores[scores_bin > 0] - 100
    iou_a = iou[np.logical_and(scores_bin > 0,
                               iou_bin > 0)]
    scores_a = scores[np.logical_and(scores_bin > 0,
                                     iou_bin > 0)]
    print(np.sum(scores_a == 0))
    iou_o = iou[np.logical_or(scores_bin > 0,
                              iou_bin > 0)]
    scores_o = scores[np.logical_or(scores_bin > 0,
                                    iou_bin > 0)]
    print(np.min(iou_a), np.max(iou_a))
    print(np.min(iou_o), np.max(iou_o))
    print(np.min(scores_a), np.max(scores_a))
    print(np.min(scores_o), np.max(scores_o))


    import seaborn as sns
    sns.set(style="ticks", context="talk")
    # sns.set_style("darkgrid")
    # plt.style.use("dark_background")
    # sns.set(color_codes=True)
    sns.set(rc={'figure.figsize':(25, 25)})
    plt.style.use("dark_background")
    ax = sns.regplot(iou_a, scores_a, order=1, ci=68, color ='red',  scatter=False, scatter_kws={"color":"lightsteelblue", "s":0.2}, line_kws={"lw": 10})
    # ax = sns.regplot(iou_o, scores_o, order=1, ci=68, color ='blue',  scatter=False, scatter_kws={"color":"lightsteelblue", "s":0.2})
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("IoU (predicted patch vs gt patch)", fontsize=60)
    ax.set_ylabel("consensus patch score", fontsize=60)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)

    ax.scatter(iou_o, scores_o, s=1, alpha=1.0, c="cyan")
    # ax.scatter(iou_a, scores_a, s=1, alpha=1.0, c="cyan")
    plt.savefig("scatter.png")


    print("used patches: fg in gt and prediction")
    p = scipy.stats.pearsonr(iou_a, scores_a)    # Pearson's r
    s = scipy.stats.spearmanr(iou_a, scores_a)   # Spearman's rho
    k = scipy.stats.kendalltau(iou_a, scores_a)  # Kendall's tau
    print(p)
    print(s)
    print(k)

    print()
    print("used patches: fg in gt or prediction")
    p = scipy.stats.pearsonr(iou_o, scores_o)    # Pearson's r
    s = scipy.stats.spearmanr(iou_o, scores_o)   # Spearman's rho
    k = scipy.stats.kendalltau(iou_o, scores_o)  # Kendall's tau
    print(p)
    print(s)
    print(k)
