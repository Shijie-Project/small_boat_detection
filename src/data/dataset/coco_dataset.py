"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os.path as osp

import faster_coco_eval
import faster_coco_eval.core.mask as coco_mask
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor


torchvision.disable_beta_transforms_warning()
faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None

__all__ = ["CocoDetection"]


@register()
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        img_folder = osp.expanduser(img_folder)
        ann_file = osp.expanduser(ann_file)
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes", spatial_size=image.size[::-1])

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        if hasattr(self, "_preset") and self._preset is not None:
            s += f" preset:\n   {repr(self._preset)}"
        return s

    @property
    def categories(self):
        return self.coco.dataset["categories"]

    @property
    def category2name(self):
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def name2category(self):
        return {cat["name"]: cat["id"] for cat in self.categories}

    @property
    def category2label(self):
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat["id"] for i, cat in enumerate(self.categories)}

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1


@register()
class SingleCocoDetection(CocoDetection):
    def __init__(
        self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False, class_name="ship"
    ):
        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
        self.filtered_ids = self.filter_imgs(class_name)

    def filter_imgs(self, class_name):
        assert class_name in self.name2category

        class_id = self.name2category[class_name]

        max_num = 0
        filtered_ids = []
        # box_areas = {}
        for img_id in self.ids:
            target = self._load_target(img_id)
            if all(t["category_id"] == class_id for t in target):
                filtered_ids.append(img_id)
                max_num = max(max_num, len(target))

            # for t in target:
            #     area = t.get("area", None)
            # if area is not None:
            #     category_name = self.category2name[t["category_id"]]
            # if category_name not in box_areas:
            #     box_areas[category_name] = []
            # box_areas[category_name].append(math.sqrt(area))
        print(
            f"Filtered {len(self.ids)} images to {len(filtered_ids)} images with only class: {class_name}. MaxNum={max_num}."
        )

        # n_classes = len(box_areas)
        # n_cols = 4  # 每行放几个 subplot，你可以改
        # n_rows = (n_classes + n_cols - 1) // n_cols
        #
        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        # axes = axes.flatten()  # 方便按一维索引访问
        #
        # for i, (cls, sizes) in enumerate(box_areas.items()):
        #     ax = axes[i]
        #     sns.histplot(sizes, bins=50, kde=False, color='skyblue', ax=ax)
        #     ax.set_xlabel("Instances' sizes")
        #     ax.set_ylabel("Instance Count")
        #     ax.set_title(f"Class: {cls}")
        #
        # # 如果 subplot 数量多于类的数量，隐藏多余的
        # for j in range(i + 1, len(axes)):
        #     axes[j].set_visible(False)
        #
        # plt.tight_layout()
        # plt.savefig("hist.png")

        return filtered_ids

    def __getitem__(self, idx):
        old_idx = self.filtered_ids[idx]
        return super().__getitem__(old_idx)

    def __len__(self):
        return len(self.filtered_ids)


@register()
class CustomDataset(CocoDetection):
    AITOD_MAP = {
        "airplane": 0,
        "bridge": 1,
        "person": 6,
        "ship": 3,
        "storage-tank": 2,
        "swimming-pool": 4,
        "vehicle": 5,
        "wind-mill": 7,
    }
    CUSTOM_MAP = {
        "10m-12m": 0,
        "12m-15m": 1,
        "15m-20m": 2,
        "5m-8m": 3,
        "8m-10m": 4,
        "<5m": 5,
        ">20m": 6,
        "Boat": 12,
        "moderate": 7,
        "none": 8,
        "ship": 9,
        "strong": 10,
        "weak": 11,
    }

    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        train_ratio=None,
        val_ratio=None,
        seed=42,
    ):
        if train_ratio is not None and val_ratio is not None:
            raise ValueError("为了保证逻辑严密，train_ratio 和 val_ratio 不能同时设置，请只设置其中一个或都不设置。")

        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)

        num_samples = len(self.ids)
        indices = np.arange(num_samples)

        # 使用固定种子进行打乱，确保每次运行、不同 split 得到的划分是完全一致的
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        # 3. 计算划分边界
        # 优先级：如果设置了 train_ratio，则取前 N 个；如果设置了 val_ratio，则取后 M 个
        if train_ratio is not None:
            assert 0 < train_ratio <= 1.0
            split_idx = int(num_samples * train_ratio)
            self.subset_indices = indices[:split_idx]
        elif val_ratio is not None:
            assert 0 < val_ratio <= 1.0
            split_idx = int(num_samples * (1 - val_ratio))
            self.subset_indices = indices[split_idx:]
        else:
            # 都不设置，则使用全量数据
            self.subset_indices = indices

        # 更新当前 Dataset 实际持有的 ID 列表
        self.ids = [self.ids[i] for i in self.subset_indices]

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        for t in target:
            t["category_id"] = self.AITOD_MAP["ship"]

        target = {"image_id": image_id, "annotations": target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes", spatial_size=image.size[::-1])

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get("category2label", None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
