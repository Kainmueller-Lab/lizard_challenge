import argparse
import glob
import os
import random
import sys
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="in_dir", required=True)
    parser.add_argument('-o', dest="out_dir", required=True)
    parser.add_argument('-f', dest="out_format", default="hdf")
    args = parser.parse_args()

    trainD_f1 = os.path.join(args.out_dir, "train_fold1")
    trainD_f2 = os.path.join(args.out_dir, "train_fold2")
    trainD_f3 = os.path.join(args.out_dir, "train_fold3")
    trainD_f12 = os.path.join(args.out_dir, "train_folds12")
    trainD_f13 = os.path.join(args.out_dir, "train_folds13")
    trainD_f23 = os.path.join(args.out_dir, "train_folds23")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(trainD_f1, exist_ok=True)
    os.makedirs(trainD_f2, exist_ok=True)
    os.makedirs(trainD_f3, exist_ok=True)
    os.makedirs(trainD_f12, exist_ok=True)
    os.makedirs(trainD_f13, exist_ok=True)
    os.makedirs(trainD_f23, exist_ok=True)

    fmt = "." + args.out_format
    trainFls = list(map(os.path.basename,
                        glob.glob(os.path.join(args.in_dir, "*" + fmt))))
    random.shuffle(trainFls)
    numFls3 = len(trainFls)//3

    if args.out_format == "hdf":
        copy_func = shutil.copy2
    elif args.out_format == "zarr":
        copy_func = shutil.copytree

    for fl in trainFls[:numFls3]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f1, fl))
    for fl in trainFls[:numFls3]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f12, fl))
    for fl in trainFls[:numFls3]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f13, fl))

    for fl in trainFls[numFls3:2*numFls3]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f2, fl))
    for fl in trainFls[numFls3:2*numFls3]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f12, fl))
    for fl in trainFls[numFls3:2*numFls3]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f23, fl))

    for fl in trainFls[2*numFls3:]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f3, fl))
    for fl in trainFls[2*numFls3:]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f13, fl))
    for fl in trainFls[2*numFls3:]:
        copy_func(os.path.join(args.in_dir, fl),
                  os.path.join(trainD_f23, fl))

if __name__ == "__main__":
    main()
