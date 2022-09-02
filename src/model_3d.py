from torchvision.models import resnet50
from torch import nn


def load_model(img_size):
    model = resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(img_size, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512 * 4, 10)
    return model

if __name__ == '__main__':
    load_model(256)
