import torchvision.datasets as vdatasets
import torchvision.transforms as trans
import torch.utils.data as tdata
import os
import cv2
import utils.cv as ucv


def get_dataset(config: ucv.config.ConfigObject):
    if config.dataset == "mnist":
        if config.trn_preprocess is None:
            config.trn_preprocess = trans.Compose([trans.ToTensor(), trans.Normalize(mean=[0.485], std=[0.229])])
        if config.val_preprocess is None:
            config.val_preprocess = trans.Compose([trans.ToTensor(), trans.Normalize(mean=[0.485], std=[0.229])])
        trn_set = vdatasets.MNIST(root=f"./datasets/mnist", train=True, transform=config.trn_preprocess, download=True)
        val_set = vdatasets.MNIST(root=f"./datasets/mnist", train=False, transform=config.val_preprocess, download=True)
        return trn_set, val_set
    elif config.dataset == "cifar10":
        if config.trn_preprocess is None:
            config.trn_preprocess = trans.Compose([
                trans.Resize((224, 224)),
                trans.RandomHorizontalFlip(p=0.5),
                trans.ToTensor(),
                trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        if config.val_preprocess is None:
            config.val_preprocess = trans.Compose([
                trans.ToTensor(),
                trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        trn_set = vdatasets.CIFAR10(root=f"./datasets/cifar10", train=True, transform=config.trn_preprocess, download=True)
        val_set = vdatasets.CIFAR10(root=f"./datasets/cifar10", train=False, transform=config.val_preprocess, download=True)
        return trn_set, val_set
    elif config.dataset == "cifar100":
        if config.trn_preprocess is None:
            config.trn_preprocess = trans.Compose([
                trans.Resize((224, 224)),
                trans.RandomHorizontalFlip(p=0.5),
                trans.ToTensor(),
                trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        if config.val_preprocess is None:
            config.val_preprocess = trans.Compose([
                trans.ToTensor(),
                trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        trn_set = vdatasets.CIFAR100(root=f"./datasets/cifar100", train=True, transform=config.trn_preprocess, download=True)
        val_set = vdatasets.CIFAR100(root=f"./datasets/cifar100", train=False, transform=config.val_preprocess, download=True)
        return trn_set, val_set
    elif config.dataset == "license_plate":
        pass
    else:
        assert config.trn_preprocess is not None
        assert config.val_preprocess is not None
        trn_set = vdatasets.ImageFolder(root=f"./datasets/{config.dataset}/train", transform=config.trn_preprocess)
        val_set = vdatasets.ImageFolder(root=f"./datasets/{config.dataset}/validation", transform=config.val_preprocess)
        return trn_set, val_set


class LprDataset(tdata.Dataset):
    def __init__(self, image_dir, image_preprocess):
        self.image_paths = [f"{image_dir}/{image_name}" for image_name in os.listdir(image_dir)]
        self.provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        self.alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
        self.ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.image
