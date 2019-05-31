import numpy as np
import cv2
import sys
import os

def main(argv):
    if len(argv) != 2:
        print("Usage: python evaluate.py estimations.npy img_names.txt")
        exit()

    with open(argv[1], "r") as f:
        files = f.readlines()

    estimations = np.load(argv[0])

    acc = 0
    for i, file in enumerate(files):
        path = os.path.join(os.getcwd(), file[:-1])
        cur = cv2.imread(path,cv2.IMREAD_COLOR)[:,:,::-1].reshape(-1).astype(np.int64)
        est = estimations[i].reshape(-1).astype(np.int64)

        cur_acc = (np.abs(cur - est) < 12).sum() / cur.shape[0]
        acc += cur_acc
    acc /= len(files)
    print(f"{acc:.2f}/1.00")


if __name__ == "__main__":
    main(sys.argv[1:])
