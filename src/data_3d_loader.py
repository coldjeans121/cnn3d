from torch.utils.data import Dataset
import numpy as np
import cv2
import h5py
import os
import csv
import random
from glob import glob
from torchvision import transforms


class Data3DLoader(Dataset):
    def __init__(self, data, data_path="F:\\Data\\mnist_3d", mode='train', img_size=256, off=0):
        # data_path="F:\\Data\\mnist_3d", mode='train'
        self.off = off
        self.mode = mode
        self.label = None
        if mode == 'train':
            self.label = {r['ID']: r['label'] for r in csv.DictReader(open(os.path.join(data_path, 'train.csv')))}
        self.data = data
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5]*img_size,
                                                                  std=[0.225]*img_size)
                                             ])

    @staticmethod
    def viz_img(img_3d):
        # img_3d *= 2
        print(np.sum(img_3d != 0))
        show_img = np.concatenate([np.sum(img_3d, axis=0), np.sum(img_3d, axis=1), np.sum(img_3d, axis=2)], axis=1)
        cv2.imshow("show_img1", show_img.astype(np.uint8))
        cv2.waitKey()

    @staticmethod
    def rotation(a, b, c, dots):
        mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        m = np.dot(np.dot(mx, my), mz)
        dots = np.dot(dots, m.T)
        return dots

    def random_rotation(self, data):
        # 45 -np.pi/4
        # print("random", data.shape)
        x, y, z = np.random.rand(3)*np.pi*2
        return self.rotation(x, y, z, data)

    def data2img(self, data):
        w = 1
        index = np.array(data)
        index = np.array((index + 1) * (self.img_size//2), dtype=int)
        img = np.zeros((self.img_size, self.img_size, self.img_size), dtype=float)
        for i in index:
            x, y, z = i
            img[x-w:x+w, y-w:y+w, z-w:z+w] += 1
        # print(np.max(img))
        img /= np.max(img)
        img *= 255
        return img.astype(np.uint8)

    def __getitem__(self, index):
        data = self.data[str(index + self.off)]
        label = index + self.off
        if self.mode == 'train':
            # data = self.random_rotation(data)
            label = self.label[str(index)]
            label = np.array(label).astype(int)
        # img = self.data2img(data)
        # img = self.transform(img)
        return str(index + self.off), data, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    test_h5 = h5py.File(os.path.join("F:\\Data\\mnist_3d", '%s.h5' % 'train'), 'r')
    dl = Data3DLoader(test_h5, off=0, mode='train')
    print(len(dl))
    for i in range(10):
        os.makedirs("G:\\Data\\mnist_3d\\data_np\\train\\%d" % i, exist_ok=True)
    for index, dd, gt in dl:
        iimg = dd
        print(index, iimg.shape, gt)
        np.save("G:\\Data\\mnist_3d\\data_np\\train\\%d\\%05d" % (gt, int(index)), dd)
        # dl.viz_img(iimg)
