import torch
import torchvision.models as Models
from torchvision.models.utils import load_state_dict_from_url
from collections import OrderedDict


model_names = ["resnet18", "alexnet", 'resnet50']
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def build_model(name, num_classes, pretrained=False):
    if name not in model_names:
        print("Mode {} does not supported!"
              "Choose a model from res18, res50, alexnet!!!\n".format(name))
        exit(-1)

    if name == "resnet18":
        model = Models.resnet18(num_classes=num_classes)
    elif name == "resnet50":
        model = Models.resnet50(num_classes=num_classes)
    elif name == "alexnet":
        model = Models.alexnet(num_classes=num_classes)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "fc" not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Load from pretrained model done.")
    print("Build {} successfully!".format(name))
    return model