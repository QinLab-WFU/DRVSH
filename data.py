import configparser
import os.path as osp
import pickle
import platform
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

_CLASS_NUM = {
    "cifar": 10,
    "flickr": 38,
    "flickr25k": 38,
    "25k": 38,
    "nuswide": 21,
    "nus": 21,
    "coco": 80,
    "imagenet": 100,
}

_TOPK = {
    "cifar": None,
    "flickr": None,
    "flickr25k": None,
    "25k": None,
    "nuswide": 5000,
    "nus": 5000,
    "coco": None,
    "imagenet": 1000,
}

def get_class_num(name):
    key = name.lower()
    if key not in _CLASS_NUM:
        raise KeyError(f"Unknown dataset: {name}. Available: {sorted(_CLASS_NUM)}")
    return _CLASS_NUM[key]

def get_topk(name):
    key = name.lower()
    if key not in _TOPK:
        raise KeyError(f"Unknown dataset: {name}. Available: {sorted(_TOPK)}")
    return _TOPK[key]

def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return np.array(lines)

def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.CenterCrop(crop_size)]
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

class BaseDataset:
    def __init__(self, name, txt_root, img_root, verbose=True):
        self.name = name
        self.txt_root = txt_root
        self.img_root = img_root
        self.train_txt = osp.join(txt_root, "train.txt")
        self.query_txt = osp.join(txt_root, "query.txt")
        self.dbase_txt = osp.join(txt_root, "dbase.txt")
        self.check_before_run()
        self.train = self.process(self.train_txt)
        self.query = self.process(self.query_txt)
        self.dbase = self.process(self.dbase_txt)
        self.set_img_abspath()
        if verbose:
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics()

    def check_before_run(self):
        for txt in [self.train_txt, self.query_txt, self.dbase_txt]:
            if not osp.exists(txt):
                raise RuntimeError(f"'{txt}' is not available")

    def process(self, txt_path):
        imgs, labs = [], []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                imgs.append(parts[0])
                labs.append(parts[1:])
        imgs = np.array(imgs)
        labs = np.array(labs, dtype=np.float32)
        return imgs, labs

    def set_img_abspath(self):
        for split in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, split)
            imgs = np.array([osp.join(self.img_root, str(img)) for img in imgs])
            setattr(self, split, (imgs, labs))

    @staticmethod
    def get_imagedata_info(data):
        labs = data[1]
        n_cids = int((labs.sum(axis=0) > 0).sum())
        n_imgs = len(data[0])
        return n_cids, n_imgs

    def print_dataset_statistics(self):
        n_train_cids, n_train_imgs = self.get_imagedata_info(self.train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(self.query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(self.dbase)
        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")

class COCO(BaseDataset):
    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    def set_img_abspath(self):
        for split in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, split)
            img_paths = []
            for img_name in imgs:
                img_id = img_name.split('.')[0]
                # 假设图片在 images/train2017 和 images/val2017 下
                train_path = osp.join(self.img_root, "images", "train2017", f"{img_id}.jpg")
                val_path = osp.join(self.img_root, "images", "val2017", f"{img_id}.jpg")
                if osp.exists(train_path):
                    img_paths.append(train_path)
                elif osp.exists(val_path):
                    img_paths.append(val_path)
                else:
                    # 兜底：尝试根目录
                    root_path = osp.join(self.img_root, img_name)
                    if osp.exists(root_path):
                        img_paths.append(root_path)
                    else:
                        img_paths.append(osp.join(self.img_root, img_name))
                        print(f"警告: COCO 找不到图片 {img_name}")
            imgs = np.array(img_paths)
            setattr(self, split, (imgs, labs))

class CIFAR(BaseDataset):
    def __init__(self, name, txt_root, img_root, verbose=True):
        super().__init__(name, txt_root, img_root, verbose)

    @staticmethod
    def unpickle(file):
        with open(file, "rb") as fo:
            dic = pickle.load(fo, encoding="latin1")
        return dic

    def set_img_abspath(self):
        data_list = [f"data_batch_{x}" for x in range(1, 6)] + ["test_batch"]
        imgs = []
        for x in data_list:
            data = self.unpickle(osp.join(self.img_root, x))
            imgs.append(data["data"])
        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        for split in ["train", "query", "dbase"]:
            _imgs, _labs = getattr(self, split)
            idxes = [int(x.replace(".png", "")) for x in _imgs]
            setattr(self, split, (imgs[idxes], _labs))

_DS_FACTORY = {
    "cifar": CIFAR,
    "nuswide": BaseDataset,
    "nus": BaseDataset,
    "flickr": BaseDataset,
    "flickr25k": BaseDataset,
    "25k": BaseDataset,
    "coco": COCO,
    "imagenet": BaseDataset,
}

def init_dataset(name, root, verbose=True):
    name = name.lower()
    if name not in _DS_FACTORY:
        raise KeyError(f"Invalid dataset: {name}. Available: {sorted(_DS_FACTORY.keys())}")

    txt_root = osp.join(root, name)

    # 对 flickr 和 nuswide 特殊处理：图片在 images/ 子目录下
    if name in ["flickr", "nuswide"]:
        img_root = osp.join(root, name, "images")
    else:
        img_root = osp.join(root, name)

    return _DS_FACTORY[name](name, txt_root, img_root, verbose=verbose)

class ImagePathDataset(Dataset):
    def __init__(self, data):
        self.imgs, self.labels = data

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 确保返回的是字符串路径
        return str(self.imgs[idx]), self.labels[idx], idx

    def get_all_labels(self):
        return torch.from_numpy(self.labels).float()

def collate_path_batch(batch):
    image_paths, labels, indices = zip(*batch)
    labels = torch.from_numpy(np.stack(labels)).float()
    indices = torch.tensor(indices, dtype=torch.long)
    return list(image_paths), labels, indices

def build_loaders(name, root, batch_size=4, num_workers=4, drop_last=True):
    data = init_dataset(name, root)
    train_loader = DataLoader(
        ImagePathDataset(data.train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_path_batch,
        pin_memory=False,
    )
    query_loader = DataLoader(
        ImagePathDataset(data.query),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_path_batch,
        pin_memory=False,
        generator=torch.Generator(),
    )
    dbase_loader = DataLoader(
        ImagePathDataset(data.dbase),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_path_batch,
        pin_memory=False,
        generator=torch.Generator(),
    )
    return train_loader, query_loader, dbase_loader