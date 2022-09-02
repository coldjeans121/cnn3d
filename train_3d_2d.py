import torch
import h5py
import os
import cv2
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data_rotatoin_loader import RotationLoader
from src.resnet_3d import resnet18
import torch.optim as optim


def main():
    pre_weight = torch.load("F:\\Data\\model\\kinetics_resnet_18_RGB_16_best.pth")
    print(pre_weight['state_dict'].keys())
    rename_weight = {}
    for key in pre_weight['state_dict']:
        if 'fc' in key:
            continue
        rename_weight[key.replace('module.', '')] = pre_weight['state_dict'][key]
        #
        # print(key, pre_weight['state_dict'][key].shape)
    model = resnet18(num_classes=3)
    init_weight = model.state_dict()
    init_weight.update(rename_weight)
    model.load_state_dict(init_weight)
    model = model.cuda()
    print(model)

    train_dl = RotationLoader(mode='train', img_size=128)
    train_loader = DataLoader(dataset=train_dl, batch_size=2, num_workers=8, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    val_dl = RotationLoader(mode='val', img_size=128)
    val_loader = DataLoader(dataset=val_dl, batch_size=2, num_workers=8, shuffle=True)

    for epoch in range(50):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (img, label, gt) in enumerate(pbar):
            img = img.unsqueeze(1)
            img = img.repeat(1, 3, 1, 1, 1)
            optimizer.zero_grad()
            pred = model(img.cuda())
            distance = gt.cuda() - pred
            loss = torch.sum(torch.multiply(distance, distance))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': '%.4f' % loss})

        model_save_path = os.path.join("F:\\Data\\mnist_3d", "exp5_resnet18_3d_%d.pth" % epoch)
        torch.save({'weight': model.state_dict()}, model_save_path)

        val_loss = 0
        with torch.no_grad():
            for i, (img, label, gt) in enumerate(val_loader):
                img = img.unsqueeze(1)
                pred = model(img.cuda())
                distance = gt.cuda() - pred
                val_loss += torch.sum(torch.multiply(distance, distance))
        print("#"*100)
        print("[%d] validation loss : %.4f" % (epoch, val_loss/len(val_loader)), end=' ')
        print("#"*100)



def inference():
    pre_weight = torch.load("F:\\Data\\mnist_3d\\exp4_resnet18_3d_9.pth")

    model = resnet18(num_classes=3)
    model.conv1 = torch.nn.Conv3d(1,
                                  64,
                                  kernel_size=7,
                                  stride=(1, 1, 1),
                                  padding=(3, 3, 3),
                                  bias=False)
    # init_weight = model.state_dict()
    # init_weight.update(pre_weight['weight'])
    # model.load_state_dict(init_weight)
    model = model.cuda()
  # default as 0.0001
    val_dl = RotationLoader(mode='val', img_size=128)
    val_loader = DataLoader(dataset=val_dl, batch_size=1, num_workers=8, shuffle=True)

    for epoch in range(50):
        model.train()
        for i, (img, label, gt) in enumerate(val_loader):
            img = img.unsqueeze(1)
            pred = model(img.cuda())
            print(pred[0], gt[0])


if __name__ == '__main__':
    main()


