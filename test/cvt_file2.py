import cv2
import os
import h5py
import csv
import numpy as np
import torch
from matplotlib import pyplot as plt
from src.data_3d_loader import Data3DLoader


def main():
    train_h5 = h5py.File(os.path.join("F:\\Data\\mnist_3d", '%s.h5' % 'train'), 'r')
    train_dl = Data3DLoader(train_h5, mode='train', img_size=28)

    test_h5 = h5py.File(os.path.join("F:\\Data\\mnist_3d", '%s.h5' % 'test'), 'r')
    test_dl = Data3DLoader(test_h5, mode='test', img_size=28, off=50000)

    [os.makedirs("F:\\Data\\mnist_3d\\data_gray\\train\\%d" % i, exist_ok=True) for i in range(10)]
    for idx, dd, gt in train_dl:
        img = dd
        print(idx, gt, img.shape)
        torch.save(img, 'F:\\Data\\mnist_3d\\data_gray\\train\\%s\\%s.pt' % (gt, idx))
        show = np.array(img[:, 10, :] * 50 + 112, dtype=np.uint8)
        cv2.imshow("test", show)
        cv2.waitKey(1)
        # for i in range(28):
        #     show = np.array(img[:, i, :]*50+112, dtype=np.uint8)
        #     cv2.imshow("test", show)
        #     cv2.waitKey(1)

def mai2():

    test_h5 = h5py.File(os.path.join("F:\\Data\\mnist_3d", '%s.h5' % 'test'), 'r')
    test_dl = Data3DLoader(test_h5, mode='test', img_size=28, off=50000)

    [os.makedirs("F:\\Data\\mnist_3d\\data_bin\\test", exist_ok=True) for i in range(10)]
    for idx, dd, gt in test_dl:
        img = dd
        print(idx, gt, img.shape)
        torch.save(img, 'F:\\Data\\mnist_3d\\data_bin\\test\\%s.pt' % idx)
        show = np.array(img[:, 10, :] * 50 + 112, dtype=np.uint8)
        cv2.imshow("test", show)
        cv2.waitKey(1)
        # for i in range(28):
        #     show = np.array(img[:, i, :]*50+112, dtype=np.uint8)
        #     cv2.imshow("test", show)
        #     cv2.waitKey(1)


def test():
    path = "F:\\Data\\mnist_3d\\train"
    for cat in range(10):
        img_list = os.listdir(os.path.join(path, "%d" % cat))
        for img_l in img_list[:100]:
            if 'pt' not in img_l:
                continue
            img = torch.load(os.path.join(path, '%d' % cat, img_l))
            print(cat, img_l, img.shape)
            show = np.array(img[:, 10, :] * 50 + 112, dtype=np.uint8)
            cv2.imshow("test", show)
            cv2.waitKey(1)
            # for i in range(28):
            #     show = np.array(img[:, i, :]*50+112, dtype=np.uint8)
            #     cv2.imshow("test", show)
            #     cv2.waitKey(1)

if __name__ == '__main__':
    mai2()
