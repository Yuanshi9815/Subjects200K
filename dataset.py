import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import os
import json
import requests
import random
import io

BASE_URL = "https://databucket2465.s3.ap-southeast-1.amazonaws.com"


def get_file(path):
    # return the file object
    file_path = os.path.join(BASE_URL, path)
    response = requests.get(file_path)
    if response.status_code == 200:
        io_file = io.BytesIO(response.content)
        return io_file
    else:
        raise Exception(f"Failed to download file {file_path}")


class Split1(Dataset):
    def __init__(
        self,
        path: str,
        condition_max_size: int = 512,
        enable_filter: bool = True,
    ):
        self.base_path = path
        self.enable_filter = enable_filter
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, "id_meta.json")):
            cloud_file = get_file("output/id_meta.json")
            with open(os.path.join(path, "id_meta.json"), "wb") as f:
                f.write(cloud_file.read())
        with open(os.path.join(path, "id_meta.json")) as f:
            self.meta_file = json.load(f)
        self.data_items = self.get_data_items(path)
        self.condition_max_size = condition_max_size
        self.to_tensor = T.ToTensor()

    def get_data_items(self, path):
        data_items = []
        for idx in self.meta_file:
            group_idx = "/".join(idx.split("/")[:-1])
            item_idx = idx.split("/")[-1].split(".")[0]
            valid_c = self.meta_file[idx]
            valid = (
                valid_c["compositeStructure"] > 4
                and valid_c["objectConsistency"] > 4
                and valid_c["imageQuality"] > 4
            )
            if not valid and self.enable_filter:
                continue
            data_items.append((group_idx, item_idx))
        return data_items

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        img_size = 512
        padding = 8

        sample_meta = self.data_items[idx]

        image_path = os.path.join(
            self.base_path, sample_meta[0], f"{sample_meta[1]}.jpg"
        )
        meta_path = os.path.join(self.base_path, sample_meta[0], f"meta.json")
        if not os.path.exists(image_path):
            if not os.path.exists(os.path.join(self.base_path, sample_meta[0])):
                os.makedirs(os.path.join(self.base_path, sample_meta[0]))
            cloud_file = get_file(f"output/{sample_meta[0]}/{sample_meta[1]}.jpg")
            with open(image_path, "wb") as f:
                f.write(cloud_file.read())
        if not os.path.exists(meta_path):
            cloud_file = get_file(f"output/{sample_meta[0]}/meta.json")
            with open(meta_path, "wb") as f:
                f.write(cloud_file.read())

        image = Image.open(image_path).convert("RGB")
        with open(meta_path) as f:
            meta = json.load(f)

        left_img = image.crop(
            (padding, padding, img_size + padding, img_size + padding)
        )
        right_img = image.crop(
            (
                img_size + padding * 2,
                padding,
                img_size * 2 + padding * 2,
                img_size + padding,
            )
        )
        descriptions = meta["description"]["scenes_descriptions"] + [
            meta["description"]["studio_description"]
        ]
        description1 = descriptions[int(sample_meta[1].split("_")[0])]
        description2 = descriptions[int(sample_meta[1].split("_")[1])]

        # image_pair = cat images
        image_pair = Image.new("RGB", (img_size * 2, img_size))
        image_pair.paste(left_img, (0, 0))
        image_pair.paste(right_img, (img_size, 0))

        return {
            "image1": left_img,
            "image2": right_img,
            "description1": description1,
            "description2": description2,
            "instance": meta["instance"],
            "image_pair": image_pair,
        }


class Split2(Dataset):
    def __init__(
        self,
        path: str,
        condition_max_size: int = 512,
        enable_filter: bool = True,
    ):
        self.base_path = path
        self.enable_filter = enable_filter

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, "id_meta.json")):
            cloud_file = get_file("output_new/id_meta_new.json")
            with open(os.path.join(path, "id_meta.json"), "wb") as f:
                f.write(cloud_file.read())
        with open(os.path.join(path, "id_meta.json")) as f:
            self.meta_file = json.load(f)

        self.data_items = self.get_data_items()
        self.condition_max_size = condition_max_size
        self.to_tensor = T.ToTensor()

    def get_data_items(self):
        data_items = []
        for idx in self.meta_file:
            group_idx = "/".join(idx.split("/")[:-1])
            item_idx = idx.split("/")[-1].split(".")[0]
            valid_c = self.meta_file[idx]
            valid = (
                valid_c["compositeStructure"] > 4
                and valid_c["objectConsistency"] > 4
                and valid_c["imageQuality"] > 4
            )
            if not valid and self.enable_filter:
                continue
            data_items.append((group_idx, item_idx))
        return data_items

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        img_size = 512
        padding = 8

        sample_meta = self.data_items[idx]

        image_path = os.path.join(
            self.base_path, sample_meta[0], f"{sample_meta[1]}.jpg"
        )
        meta_path = os.path.join(self.base_path, sample_meta[0], f"meta.json")
        if not os.path.exists(image_path):
            if not os.path.exists(os.path.join(self.base_path, sample_meta[0])):
                os.makedirs(os.path.join(self.base_path, sample_meta[0]))
            cloud_file = get_file(f"output_new/{sample_meta[0]}/{sample_meta[1]}.jpg")
            with open(image_path, "wb") as f:
                f.write(cloud_file.read())
        if not os.path.exists(meta_path):
            cloud_file = get_file(f"output_new/{sample_meta[0]}/meta.json")
            with open(meta_path, "wb") as f:
                f.write(cloud_file.read())

        image = Image.open(image_path).convert("RGB")
        with open(meta_path) as f:
            meta = json.load(f)

        left_img = image.crop(
            (padding, padding, img_size + padding, img_size + padding)
        )
        right_img = image.crop(
            (
                img_size + padding * 2,
                padding,
                img_size * 2 + padding * 2,
                img_size + padding,
            )
        )

        descriptions = meta["scene_descriptions"] + [meta["studio_photo_description"]]
        description1 = descriptions[int(sample_meta[1].split("_")[0])]
        description2 = descriptions[int(sample_meta[1].split("_")[1])]

        image_pair = Image.new("RGB", (img_size * 2, img_size))
        image_pair.paste(left_img, (0, 0))
        image_pair.paste(right_img, (img_size, 0))

        return {
            "image1": left_img,
            "image2": right_img,
            "description1": description1,
            "description2": description2,
            "instance": meta["brief_description"],
            "image_pair": image_pair,
        }


class Subjects200KDataset(Dataset):
    def __init__(
        self,
        base_path: str = "data",
        path_1: str = "split1",
        path_2: str = "split2",
        condition_max_size: int = 512,
        use_split1: bool = True,
        use_split2: bool = True,
        split1_bias: int = 1,  # How many times more likely to sample from split1
        enable_filter: bool = True,  # Whether to filter out invalid samples
    ):
        self.base_path = base_path
        self.use_split1 = use_split1
        self.use_split2 = use_split2
        self.enable_filter = enable_filter

        if not use_split1 and not use_split2:
            raise ValueError("At least one split must be used")

        if use_split1:
            self.iddata1_path = os.path.join(base_path, path_1)
            self.dataset1 = Split1(self.iddata1_path, condition_max_size, enable_filter)
            self.dataset1_bias = split1_bias

        if use_split2:
            self.iddata2_path = os.path.join(base_path, path_2)
            self.dataset2 = Split2(self.iddata2_path, condition_max_size, enable_filter)

    def __len__(self):
        length = 0
        if self.use_split1:
            length += len(self.dataset1) * self.dataset1_bias
        if self.use_split2:
            length += len(self.dataset2)
        return length

    def __getitem__(self, idx):
        if self.use_split1:
            split1_length = len(self.dataset1) * self.dataset1_bias
            if idx < split1_length:
                return self.dataset1[idx % len(self.dataset1)]
            idx -= split1_length

        if self.use_split2:
            return self.dataset2[idx]
