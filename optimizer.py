import torch


def build_optimizer(params, configs):
    if configs.Train.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params=params, lr=configs.Train.lr, momentum=configs.Train.momentum,
                            weight_decay=configs.Train.weight_decay)

    return optimizer