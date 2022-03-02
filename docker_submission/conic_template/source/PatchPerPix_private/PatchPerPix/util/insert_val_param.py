import argparse
import os
import sys
import glob
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input",
                        help=('experiment folder'))
    parser.add_argument('--param',
                        help=('parameter name'))
    parser.add_argument('--value',
                        help=('parameter value'))
    parser.add_argument('--position', type=int,
                        help=('parameter position (starting with one)'))

    args = parser.parse_args()

    print(args)
    if args.input[-1] == "/":
        args.input = args.input[:-1]
    if os.path.basename(args.input) in ['val', 'test']:
        args.input = os.path.dirname(args.input)

    print("input dir: {}".format(args.input))

    subdirs = ['val/instanced', 'val/evaluated',
               'test/instanced', 'test/evaluated']

    cwd = os.getcwd()
    for sd in subdirs:
        os.chdir(os.path.join(cwd, args.input))
        try:
            os.chdir(sd)
        except FileNotFoundError as e:
            print("Folder doesnt exist", e)
            continue
        except NotADirectoryError as e:
            print("Not a directory", e)
            continue

        print("doing {}".format(sd))
        recurse_dir_tree(args, ".", 0)


def recurse_dir_tree(args, cur_dir_name, cur_level):
    fls = os.listdir(cur_dir_name)
    if cur_level == args.position:
        nm = args.param + "_" + args.value.replace(".", "_")
        print("new param name: {}".format(nm))

        for fl in fls:
            if nm in fl:
                print("new param dir already exists!")
                return
        os.makedirs(os.path.join(cur_dir_name, nm))
        for fl in fls:
            shutil.move(os.path.join(cur_dir_name, fl),
                        os.path.join(cur_dir_name, nm))
        return
    hit_bottom = True
    for fl in fls:
        if os.path.isdir(os.path.join(cur_dir_name, fl)):
            recurse_dir_tree(args, os.path.join(cur_dir_name, fl), cur_level+1)
            hit_bottom = False
    if hit_bottom:
        print("hit bottom of directory tree, position too large?")

if __name__ == "__main__":
    main()
