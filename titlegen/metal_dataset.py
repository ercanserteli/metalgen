import json
import os

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms


class MetalAlbumsResnetDataset(Dataset):
    def __init__(self, titles_path, resnet_path, device="cuda"):
        with open(titles_path, "r") as f:
            albums = json.load(f)

        self.album_ids = [a[0] for a in albums]
        self.album_titles = [a[1].lower() for a in albums]

        self.album_resnets = torch.from_numpy(np.load(resnet_path)).to(torch.float32).to(device)

        self.int2char = None
        self.char2int = None
        self.char_count = 0
        total_chars = tuple(set("".join(self.album_titles)))
        int2char = dict([(i+2, c) for (i, c) in enumerate(total_chars)])
        int2char[1] = "[END]"
        self.set_int2char(int2char)

    def set_int2char(self, int2char):
        self.int2char = int2char
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.char_count = len(self.char2int)

    def __len__(self):
        return len(self.album_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        int_title = [self.char2int[c] for c in self.album_titles[idx]] + [1]
        # one_hot_title = one_hot_encode(np.array(int_title), self.char_count)
        sample = {"resnet": self.album_resnets[idx, :], "title": torch.tensor(int_title, dtype=torch.long, device="cuda")}
        return sample


class MetalAlbumsDataset(Dataset):
    def __init__(self, json_path, jpg_dir):
        with open(json_path, "r") as f:
            albums = json.load(f)

        self.album_ids = [a[0] for a in albums]
        self.jpg_dir = jpg_dir

        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]
        )

    def __len__(self):
        return len(self.album_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.jpg_dir, self.album_ids[idx] + ".jpg")
        image = self.transform(Image.open(img_path)).to("cuda")
        sample = {"image": image, "id": self.album_ids[idx]}
        return sample


def pack_collate_images(batch):
    id_data = [item["id"] for item in batch]

    image_data = torch.Tensor(len(batch), *batch[0]["image"].shape).to("cuda")
    torch.cat([item["image"].unsqueeze(0) for item in batch], out=image_data)
    return image_data, id_data


def pack_collate_resnets(batch):
    title_data = [item["title"] for item in batch]

    lengths = [t.size(0) for t in title_data]
    padded_title_data = pad_sequence(title_data)

    resnet_data = torch.Tensor(len(batch), *batch[0]["resnet"].shape).to("cuda")
    for i, item in enumerate(batch):
        resnet_data[i, :] = item["resnet"]
    return resnet_data, padded_title_data, lengths
