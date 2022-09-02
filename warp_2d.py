import torch
import h5py
import os
import cv2
import numpy as np
import torch

from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from src.data_3d_loader import Data3DLoader
from src.model_3d_2d import resnet18
import torch.optim as optim


def main():
    pre_weight = torch.load("F:\\Data\\model\\kinetics_resnet_18_RGB_16_best.pth")
    print(pre_weight['state_dict'].keys())
    rename_weight = {}
    for key in pre_weight['state_dict']:
        if 'fc' in key:
            continue
        rename_weight[key.replace('module.', '')] = pre_weight['state_dict'][key]
    model = resnet18()
    init_weight = model.state_dict()
    init_weight.update(rename_weight)
    model.load_state_dict(init_weight)
    model.conv1 = torch.nn.Conv3d(1,
                                  64,
                                  kernel_size=5,
                                  stride=(1, 1, 1),
                                  padding=(3, 3, 3),
                                  bias=False)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    loss_function = CrossEntropyLoss()

    train_h5 = h5py.File(os.path.join("F:\\Data\\mnist_3d", '%s.h5' % 'train'), 'r')
    train_dl = Data3DLoader(train_h5, mode='train', img_size=32)
    train_loader = DataLoader(dataset=train_dl, batch_size=32, num_workers=0, shuffle=True)

    for epoch in range(50):
        model.train()
        for i, (_, img, gt) in enumerate(train_loader):
            img = img.unsqueeze(1)
            optimizer.zero_grad()
            pred = model(img.cuda())
            gt = torch.tensor(gt, dtype=torch.long).cuda()
            loss = loss_function(pred, gt)
            loss.backward()
            optimizer.step()
            label = torch.argmax(pred, dim=1)
            acc = torch.sum(gt == label) / len(gt)
            print("[%d] %03d loss : %.4f, acc : %.4f" % (epoch, i, loss, acc), end=' ')
            for b in range(2):
                print("(%d, %d)" % (gt[b], label[b]), end=", ")
            print("")

        model_save_path = os.path.join("F:\\Data\\mnist_3d", "exp3_resnet18_3d_%d.pth" % epoch)
        torch.save({'weight': model.state_dict()}, model_save_path)



if __name__ == '__main__':
    main()


