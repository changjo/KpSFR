"""
train and test set on WorldCup for Nie et al. (A robust and efficient framework for sports-field registration)
"""
import os.path as osp
import random
from glob import glob
from typing import Optional

import numpy as np
import skimage.segmentation as ss
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

import utils

ORIGINAL_SIZE = (1280, 720)


# def crop(
#     img: Image.Image,
#     top: int,
#     left: int,
#     height: int,
#     width: int,
# ) -> Image.Image:

#     if not _is_pil_image(img):
#         raise TypeError(f"img should be PIL Image. Got {type(img)}")

#     return img.crop((left, top, left + width, top + height))


class PitchDataset(data.Dataset):
    def __init__(self, root, crop_ratio=0.7):
        self.root = root
        self.crop_ratio = crop_ratio
        # self.videos_path = osp.join(self.root, "videos")
        self.videos_path = self.root
        self.file_list = []
        for video_path in sorted(glob(osp.join(self.videos_path, "*"))):
            self.file_list += sorted(glob(osp.join(video_path, "images", "*.jpg")))

        #     (top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(100, 100))(orig_img)
        # plot([top_left, top_right, bottom_left, bottom_right, center])

        # crop_size = (
        #     int(ORIGINAL_SIZE[0] * self.crop_ratio),
        #     int(ORIGINAL_SIZE[1] * self.crop_ratio),
        # )
        self.crop_size = (ORIGINAL_SIZE[0], ORIGINAL_SIZE[1])
        self.new_img_size = (
            int(ORIGINAL_SIZE[0] / self.crop_ratio),
            int(ORIGINAL_SIZE[1] / self.crop_ratio),
        )

        center_crop_top = int(round((self.new_img_size[1] - self.crop_size[1]) / 2.0))
        center_crop_left = int(round((self.new_img_size[0] - self.crop_size[0]) / 2.0))

        self.crop_region = {
            "top_left": (0, 0, self.crop_size[0], self.crop_size[1]),
            "top_right": (
                self.new_img_size[0] - self.crop_size[0],
                0,
                self.new_img_size[0],
                self.crop_size[1],
            ),
            "bottom_left": (
                0,
                self.new_img_size[1] - self.crop_size[1],
                self.crop_size[0],
                self.new_img_size[1],
            ),
            "bottom_right": (
                self.new_img_size[0] - self.crop_size[0],
                self.new_img_size[1] - self.crop_size[1],
                self.new_img_size[0],
                self.new_img_size[1],
            ),
            "center": (
                center_crop_left,
                center_crop_top,
                center_crop_left + self.crop_size[0],
                center_crop_top + self.crop_size[1],
            ),
        }

        self.crop_order = [
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
            "center",
        ]
        # tl = crop(img, 0, 0, crop_height, crop_width)
        # tr = crop(img, 0, image_width - crop_width, crop_height, crop_width)
        # bl = crop(img, image_height - crop_height, 0, crop_height, crop_width)
        # br = crop(
        #     img,
        #     image_height - crop_height,
        #     image_width - crop_width,
        #     crop_height,
        #     crop_width,
        # )

        self.preprocess = T.Compose(
            [
                T.Resize(size=self.new_img_size[::-1]),
                T.FiveCrop(size=ORIGINAL_SIZE[::-1]),
                T.Lambda(
                    lambda crops: [
                        T.Resize(size=ORIGINAL_SIZE[::-1])(crop) for crop in crops
                    ]
                ),
                T.Lambda(lambda crops: [T.ToTensor()(crop) for crop in crops]),
                T.Lambda(
                    lambda crops: [
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
                        for crop in crops
                    ]
                ),  # ImageNet
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        im = Image.open(filename)
        # left, upper, right, lower = 1300, 400, 1300 + 1500, 400 + 900
        # im = im.crop((left, upper, right, lower))
        # im = im.resize(ORIGINAL_SIZE)
        image = np.array(im)

        template_grid = utils.gen_template_grid()  # template grid shape (91, 3)
        warp_image = utils.infer_gen_im_partial_grid(image, template_grid)

        pil_image = Image.fromarray(warp_image)
        image_tensor_list = self.preprocess(pil_image)

        image_tensor_dict = {}
        for crop_loc, image_tensor in zip(self.crop_order, image_tensor_list):
            image_tensor_dict[crop_loc] = image_tensor

        image_resized = T.Resize(size=self.new_img_size[::-1])(pil_image)
        image_tensor_dict["image_resized"] = T.ToTensor()(image_resized)

        return image_tensor_dict, filename
