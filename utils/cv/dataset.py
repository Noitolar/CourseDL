import torchvision.datasets as vdatasets
import torchvision.transforms as trans


def get_dataset(dataset_name, trn_preprocess=None, val_preprocess=None):
    if dataset_name == "mnist":
        if trn_preprocess is None:
            trn_preprocess = trans.Compose([trans.ToTensor(), trans.Normalize(mean=[0.485], std=[0.229])])
        if val_preprocess is None:
            val_preprocess = trans.Compose([trans.ToTensor(), trans.Normalize(mean=[0.485], std=[0.229])])
        trn_set = vdatasets.MNIST(root=f"./datasets/mnist", train=True, transform=trn_preprocess, download=True)
        val_set = vdatasets.MNIST(root=f"./datasets/mnist", train=False, transform=val_preprocess, download=True)
        return trn_set, val_set
    elif dataset_name == "cifar10":
        return None, None
    elif dataset_name == "cifar100":
        return None, None
    else:
        assert trn_preprocess is not None
        assert val_preprocess is not None
        trn_set = vdatasets.ImageFolder(root=f"./datasets/{dataset_name}/train", transform=trn_preprocess)
        val_set = vdatasets.ImageFolder(root=f"./datasets/{dataset_name}/validation", transform=val_preprocess)
        return trn_set, val_set
