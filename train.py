import torch
import torch.nn as nn
from torch.utils import data
from models import build_model
from optimizer import build_optimizer
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import yaml
import easydict as Edict

def adjust_lr(optimizer, configs):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= configs.Train.lr_decay

def train_net(configs):
    model = build_model(configs.backbone, num_classes=configs.Num_Classes, pretrained=configs.Pretrained)
    #print(model)
    optimizer = build_optimizer(model.parameters(), configs)
    criterion = nn.CrossEntropyLoss()
    if configs.cuda:
        device = torch.device("cuda")
        model.to(device)
        criterion.to(device)
    if configs.img_aug:
        imgaug = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=configs.mean, std=configs.std),
                                     ])
        train_set = datasets.ImageFolder(configs.train_root, transform=imgaug)
        train_loader = data.DataLoader(train_set, batch_size=configs.Train.batch_size,
                                                   shuffle=configs.shuffle, num_workers=configs.num_workers, pin_memory=True)
    else:
        train_set = datasets.ImageFolder(configs.train_root, transform=None)
        train_loader = data.Dataloader(train_set, batch_size=configs.Train.batch_size,
                                                   shuffle=configs.shuffle, num_workers=configs.num_workers,
                                                   pin_memory=True)
    for epoch in range(configs.Train.nepochs):
        if epoch > 0 and epoch // 2 == 0:
            adjust_lr(optimizer, configs)
        for idx, (img, target) in enumerate(train_loader):
            if configs.cuda:
                device = torch.device("cuda")
                img = img.to(device)
                target = target.to(device)
            out = model(img)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("|Epoch|: {}, {}/{}, loss{}".format(epoch, idx, len(train_set) // configs.Train.batch_size, loss.item()))
        pth_path = "./weights/{}_{}.pth".format(configs.backbone, epoch)
        with open(pth_path, 'wb') as f:
            torch.save(model.state_dict(), f)
            print("Save weights to ---->{}<-----".format(pth_path))

    with open("./weights/final.pth", 'wb') as f:
        torch.save(model.state_dict(), f)
        print("Final model saved!!!")

if __name__ == "__main__":
    with open("./config.yaml") as f:
        configs = yaml.load(f)
        configs = Edict.EasyDict(configs)
    train_net(configs)