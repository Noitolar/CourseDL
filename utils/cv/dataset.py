import torchvision.datasets as vdatasets
import torchvision.transforms as trans
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
    else:
        assert config.trn_preprocess is not None
        assert config.val_preprocess is not None
        trn_set = vdatasets.ImageFolder(root=f"./datasets/{config.dataset}/train", transform=config.trn_preprocess)
        val_set = vdatasets.ImageFolder(root=f"./datasets/{config.dataset}/validation", transform=config.val_preprocess)
        return trn_set, val_set
