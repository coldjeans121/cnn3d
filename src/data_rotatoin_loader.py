from torch.utils.data import Dataset
import numpy as np
import cv2
import h5py
import os
import torch
import csv
import random
from glob import glob
from torchvision import transforms


class RotationLoader(Dataset):
    def __init__(self, data_path="G:\\Data\\mnist_3d\\data_np", mode='train', img_size=128):
        # data_path="F:\\Data\\mnist_3d", mode='train'
        self.mode = mode
        self.data, self.label = self.get_img_list(data_path)
        self.img_size = img_size
        self.transform_3d = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5]*img_size,
                                                                  std=[0.225]*img_size)
                                             ])

    def get_img_list(self, data_path):
        if self.mode == 'test':
            img_list = glob(os.path.join(data_path, 'test', '*.npy'))
            label_list = [int(img_l.split(os.sep)[-1].split('.')[0]) for img_l in img_list]
            return img_list, label_list
        else:
            img_list = []
            label_list = []
            for i in range(10):
                sub_path = os.path.join(data_path, 'train', '%d' % i)
                sub_img_list = [os.path.join(sub_path, s) for s in os.listdir(sub_path) if 'npy' in s]
                sub_img_num = len(sub_img_list)
                sub_img_list.sort()
                if self.mode == 'train':
                    sub_img_list = sub_img_list[:int(sub_img_num*0.9)]
                elif self.mode == 'val':
                    sub_img_list = sub_img_list[int(sub_img_num*0.9):]
                else:
                    raise AssertionError
                sub_label = [i for _ in range(len(sub_img_list))]
                img_list += sub_img_list
                label_list += sub_label

            return img_list, label_list

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
        x, y, z = np.random.rand(3)
        # x, y, z = 0, 0, 0
        return self.rotation(x*np.pi*2, y*np.pi*2, z*np.pi*2, data), [x, y, z]

    def cvt_data(self, data):
        data_np = np.array(data[:, 14, :], dtype=float)
        data_np -= np.min(data_np)
        data_np /= np.max(data_np)
        img = np.array(data_np*255, dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
        index = 0
        data = self.data[index]
        label = self.label[index]
        try:
            data = np.load(data)
        except Exception as e:
            print('='*100)
            print(index, e)
            print('='*100)
            index = 0
            data = self.data[index]
            data = np.load(data)
            label = self.label[index]

        data, rota = self.random_rotation(data)

        img = self.data2img(data)
        img = self.transform_3d(img)
        return img, label, np.array(rota, dtype=float)

    def __len__(self):
        return len(self.data)
# class RotationLoader(Dataset):
#     def __init__(self, data_path="F:\\Data\\mnist_3d\\data_gray", mode='train'):
#         # data_path="F:\\Data\\mnist_3d", mode='train'
#         self.mode = mode
#         self.data = self.get_img_list(data_path)
#         self.transform = transforms.Compose([transforms.ToTensor(),
#                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                   std=[0.229, 0.224, 0.225])
#                                              ])
#
#     def get_img_list(self, data_path):
#         img_list = glob(os.path.join(data_path, self.mode, '*/*.pt'))
#         return img_list
#
#     @staticmethod
#     def rotation(a, b, c, dots):
#         mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
#         my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
#         mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
#         print(dots.shape, mx.shape)
#         m = np.dot(np.dot(mx, my), mz)
#         dots = np.dot(dots, m.T)
#         return dots
#
#     def random_rotation(self, data):
#         # 45 -np.pi/4
#         x, y, z = np.random.rand(3)
#         return self.rotation(x*np.pi*2, y*np.pi*2, z*np.pi*2, data), [x, y, z]
#
#     def cvt_data(self, data):
#         data_np = np.array(data[:, 14, :], dtype=float)
#         data_np -= np.min(data_np)
#         data_np /= np.max(data_np)
#         img = np.array(data_np*255, dtype=np.uint8)
#         return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#     def __getitem__(self, index):
#         data = self.data[index]
#         data_np = np.array(torch.load(data))
#         img, label = self.random_rotation(data_np)
#         img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.data)

