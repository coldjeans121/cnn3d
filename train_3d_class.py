import torch
import h5py
import os
import cv2
import numpy as np
import torch

from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from src.data_np_loader import DataLoader as DL
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

    train_dl = DL(mode='train', img_size=64)
    train_loader = DataLoader(dataset=train_dl, batch_size=16, num_workers=8, shuffle=True)
    val_dl = DL(mode='val', img_size=64)
    val_loader = DataLoader(dataset=val_dl, batch_size=16, num_workers=8, shuffle=True)

    for epoch in range(50):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (img, gt) in enumerate(pbar):
            img = img.unsqueeze(1)
            optimizer.zero_grad()
            pred = model(img.cuda())
            gt = torch.tensor(gt, dtype=torch.long).cuda()
            loss = loss_function(pred, gt)
            loss.backward()
            optimizer.step()
            label = torch.argmax(pred, dim=1)
            acc = torch.sum(gt == label) / len(gt)
            pbar.set_postfix({'loss': '%.4f' % loss, 'acc': '%.4f' % acc})
            # print("[%d] %03d loss : %.4f, acc : %.4f" % (epoch, i, loss, acc), end=' ')
            # for b in range(2):
            #     print("(%d, %d)" % (gt[b], label[b]), end=", ")
            # print("")
        with torch.no_grad():
            true_sample_num = 0
            for i, (img, gt) in enumerate(val_loader):
                img = img.unsqueeze(1)
                pred = model(img.cuda())
                gt = torch.tensor(gt, dtype=torch.long).cuda()
                label = torch.argmax(pred, dim=1)
                true_sample_num += torch.sum(gt == label)
            print("[%d] validation acc : %.4f" % (epoch, true_sample_num/len(val_dl)), end=' ')


        model_save_path = os.path.join("F:\\Data\\mnist_3d", "exp3_resnet18_3d_%d.pth" % epoch)
        torch.save({'weight': model.state_dict()}, model_save_path)


def inference():
    pre_weight = torch.load("F:\\Data\\mnist_3d\\exp3_resnet18_3d_5.pth")
    print(pre_weight['weight'].keys())

    model = resnet18()
    model.conv1 = torch.nn.Conv3d(1,
                                  64,
                                  kernel_size=5,
                                  stride=(1, 1, 1),
                                  padding=(3, 3, 3),
                                  bias=False)
    init_weight = model.state_dict()
    init_weight.update(pre_weight['weight'])
    model.load_state_dict(init_weight)
    model = model.cuda()

    val_dl = DL(mode='test', img_size=64)
    val_loader = DataLoader(dataset=val_dl, batch_size=16, num_workers=8, shuffle=False)


    with torch.no_grad():
        true_sample_num = 0
        for i, (img, gt) in enumerate(val_loader):
            img = img.unsqueeze(1)
            pred = model(img.cuda())
            gt = torch.tensor(gt, dtype=torch.long).cuda()
            label = torch.argmax(pred, dim=1)
            true_sample_num += torch.sum(gt == label)
            b = len(gt)
            print(i, len(val_loader), gt[0], label[0])
            for k in range(b):
                with open("F:\\Data\\mnist_3d\\result.txt", 'a') as f:
                    f.write("%d, %d\n" % (gt[k], label[k]))
        print("validation acc : %.4f" % (true_sample_num / len(val_dl)), end=' ')


if __name__ == '__main__':
    inference()


