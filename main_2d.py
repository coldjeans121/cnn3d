import h5py
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from src.data_loader import DataLoader as DL
from src.model_2d import load_model
import torch.optim as optim


def train(img_size):
    train_dl = DL(mode='train', img_size=img_size)
    train_loader = DataLoader(dataset=train_dl, batch_size=128, num_workers=4, shuffle=True)
    val_dl = DL(mode='val', img_size=img_size)
    val_loader = DataLoader(dataset=val_dl, batch_size=128, num_workers=4, shuffle=True)
    print(len(train_loader), len(val_loader))
    # test_dl = Data3DLoader(mode='test', img_size=img_size)
    model = load_model(img_size=img_size)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_function = CrossEntropyLoss()

    for epoch in range(50):
        model.train()

        for i, (img, gt) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(img.cuda())
            gt = torch.tensor(gt, dtype=torch.long).cuda()
            loss = loss_function(pred, gt)
            loss.backward()
            optimizer.step()
            label = torch.argmax(pred, dim=1)
            acc = torch.sum(gt == label) / len(gt)
            print("[%d] %03d loss : %.4f, acc : %.4f" % (epoch, i, loss, acc))


        model_save_path = os.path.join("F:\\Data\\mnist_3d", "exp2_resnet23_%d.pth" % epoch)
        torch.save({'weight': model.state_dict()}, model_save_path)
        
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            total_data = 0
            for img, gt in tqdm(val_loader, total=len(val_loader)):
                pred = model(img.cuda())
                gt = torch.tensor(gt, dtype=torch.long).cuda()
                total_loss += loss_function(pred, gt)
                label = torch.argmax(pred, dim=1)
                total_acc += torch.sum(gt == label)
                total_data += len(gt)
            print("valid loss : %.4f, acc : %.4f" % (total_loss, float(total_acc/total_data)))



def inference(img_size):
    model = load_model(img_size=img_size)
    weight_dict = torch.load("F:\\Data\\mnist_3d\\exp1_resnet50_3.pth")
    model.load_state_dict(weight_dict['weight'])
    model = model.cuda()

    train_h5 = h5py.File(os.path.join("F:\\Data\\mnist_3d", '%s.h5' % 'train'), 'r')
    train_dl = Data3DLoader(train_h5, mode='train', img_size=img_size)
    train_loader = DataLoader(dataset=train_dl, batch_size=32, num_workers=0, shuffle=True)

    test_h5 = h5py.File(os.path.join("F:\\Data\\mnist_3d", '%s.h5' % 'test'), 'r')
    test_dl = Data3DLoader(test_h5, mode='test', img_size=img_size, off=50000)
    test_loader = DataLoader(dataset=test_dl, batch_size=32, num_workers=0, shuffle=False)

    model.eval()
    true_sample = 0
    total_sample = 0
    for i, (img, gt) in enumerate(train_loader):
        with torch.no_grad():
            pred = model(img.cuda())
            label = torch.argmax(pred, dim=1)
            true_sample += torch.sum(gt.cuda() == label)
            total_sample += gt.shape[0]
            print(gt, label)
            print(true_sample, total_sample, float(100*true_sample/total_sample))
    print(true_sample, total_sample)

if __name__ == '__main__':
    train(256)

