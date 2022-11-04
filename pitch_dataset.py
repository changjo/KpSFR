import os
import os.path as osp
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as ss
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

import utils


class PitchDataset(data.Dataset):
    def __init__(
        self,
        root,
        num_objects,
        sfp_finetuned: Optional[bool] = False,
    ):

        self.frame_h = 720
        self.frame_w = 1280
        self.root = root
        self.num_objects = num_objects
        # self.num_objects = 91
        self.sfp_finetuned = sfp_finetuned

        if self.sfp_finetuned:
            sfp_out_path = "SingleFramePredict_finetuned_with_normalized"
        else:
            sfp_out_path = "SingleFramePredict_with_normalized"

        # self.videos_path = osp.join(self.root, "videos")
        self.videos_path = self.root
        self.videos = []
        self.frames = {}
        self.num_frames = {}
        self.segs = {}
        self.crop_regions = {}

        for video_path in sorted(glob(osp.join(self.videos_path, "*"))):
            _video = osp.basename(video_path)
            self.videos.append(_video)
            frames = sorted(glob(osp.join(video_path, "crop", "*.jpg")))
            crop_region_file_list = sorted(glob(osp.join(video_path, "crop", "*.txt")))
            self.crop_regions[_video] = []
            for crop_region_filename in crop_region_file_list:
                with open(crop_region_filename, "r") as f:
                    self.crop_regions[_video].append([int(v) for v in f.readlines()[0].split(",")])

            self.num_frames[_video] = len(frames)
            self.frames[_video] = frames
            sfp_path = osp.join(video_path, sfp_out_path)
            self.segs[_video] = sorted(glob(osp.join(sfp_path, "*.png")))

        print(self.frames)
        print(self.segs)
        # self.anno_path = osp.join(self.root, "Annotations")
        # self.sfp_path = osp.join(self.root, sfp_out_path)

        # self.num_frames = len(self.image_file_list)
        # self.num_homographies[_video] = len(
        #     glob(osp.join(self.anno_path, _video, "*_homography.npy"))
        # )

        # self.frames = self.image_file_list

        # homographies = sorted(os.listdir(osp.join(self.anno_path, _video)))
        # self.homographies[_video] = homographies

        # self.segs = sorted(glob(osp.join(self.sfp_path, "*.png")))

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
            ]
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        _video_name = self.videos[index]
        _frames = self.frames[_video_name]
        # _homographies = self.homographies[_video_name]
        _segs = self.segs[_video_name]
        info = {}
        info["num_objects"] = self.num_objects
        info["name"] = _video_name
        info["frames"] = []
        info["num_frames"] = self.num_frames[_video_name]
        info["single_frame_path"] = self.segs[_video_name]
        info["crop_regions"] = self.crop_regions[_video_name]

        template_grid = utils.gen_template_grid()  # template grid shape (91, 3)

        image_list = []
        # homo_mat_list = []
        # dilated_hm_list = []
        # hm_list = []
        gt_seg_list = []

        for f_idx in range(self.num_frames[_video_name]):
            jpg_image = _frames[f_idx]
            # npy_matrix = _homographies[f_idx]
            png_seg = _segs[f_idx]
            info["frames"].append(jpg_image)
            im = Image.open(jpg_image)
            im = im.resize((1280, 720))
            image = np.array(im)

            # gt_h = np.load(osp.join(self.anno_path, _video_name, npy_matrix))

            sfp_seg = np.array(Image.open(png_seg).convert("P"))
            gt_seg_list.append(sfp_seg)

            # warp grid shape (91, 3)
            unigrid, warp_image = utils.infer_gen_im_whole_grid(image, template_grid)

            # Each keypoints is considered as an object
            # num_pts = warp_grid.shape[0]
            num_pts = unigrid.shape[0]
            pil_image = Image.fromarray(warp_image)

            image_tensor = self.preprocess(pil_image)
            image_list.append(image_tensor)
            # homo_mat_list.append(homo_mat)

            # By default, all keypoints belong to background
            # C*H*W, C:91, exclude background class
            # heatmaps = np.zeros(
            #     (num_pts, self.frame_h // 4, self.frame_w // 4), dtype=np.float32
            # )
            # dilated_heatmaps = np.zeros_like(heatmaps)
            # for keypts_label in range(num_pts):
            #     if np.isnan(warp_grid[keypts_label, 0]) and np.isnan(
            #         warp_grid[keypts_label, 1]
            #     ):
            #         continue
            #     px = np.rint(warp_grid[keypts_label, 0] / 4).astype(np.int32)
            #     py = np.rint(warp_grid[keypts_label, 1] / 4).astype(np.int32)
            #     cls = int(warp_grid[keypts_label, 2]) - 1
            #     if 0 <= px < (self.frame_w // 4) and 0 <= py < (self.frame_h // 4):
            #         heatmaps[cls][py, px] = warp_grid[keypts_label, 2]
            #         dilated_heatmaps[cls] = ss.expand_labels(heatmaps[cls], distance=5)

            # dilated_hm_list.append(dilated_heatmaps)
            # hm_list.append(heatmaps)

        # TODO: use full gt segmentatino info, only previous for memory management
        # dilated_hm_list = np.stack(dilated_hm_list, axis=0)  # num_frames*91*H*W
        # T, CK, H, W = dilated_hm_list.shape
        # hm_list = np.stack(hm_list, axis=0)

        # (CK:num_objects, T:num_frames, H:180, W:320)
        # target_dilated_hm_list = torch.zeros((CK, T, H, W))
        # target_hm_list = torch.zeros_like(target_dilated_hm_list)
        # cls_gt = torch.zeros((self.num_frames[_video_name], H, W))
        lookup_list = []
        # for f in range(self.num_frames):
        #     class_lables = np.ones(num_pts, dtype=np.float32) * -1
        #     # Those keypoints appears on the each frame
        #     labels = np.unique(dilated_hm_list[f])
        #     labels = labels[labels != 0]  # Remove background class
        #     for obj in labels:
        #         class_lables[int(obj) - 1] = obj

        #     for idx, obj in enumerate(class_lables):
        #         if obj != -1:
        #             target_dilated_hm = dilated_hm_list[f, int(obj) - 1].copy()
        #             target_dilated_hm[target_dilated_hm == obj] = 1
        #             target_dilated_hm_tensor = utils.to_torch(target_dilated_hm)
        #             target_dilated_hm_list[int(obj) - 1, f] = target_dilated_hm_tensor

        #             target_hm = hm_list[f, int(obj) - 1].copy()
        #             target_hm[target_hm == obj] = 1
        #             target_hm_tensor = utils.to_torch(target_hm)
        #             target_hm_list[int(obj) - 1, f] = target_hm_tensor

        #     # TODO: union of all target objects of ground truth segmentation
        #     for idx, obj in enumerate(class_lables):
        #         if obj != -1:
        #             cls_gt[target_hm_list[idx] == 1] = torch.tensor(obj).float()

        # TODO: use full single frame predict segmentatino info, only previous for memory management
        gt_seg_list = np.stack(gt_seg_list, axis=0)  # num_frames*H*W
        del_idx_list = []
        for f in range(self.num_frames[_video_name]):
            class_lables = np.ones(num_pts, dtype=np.float32) * -1
            # Those keypoints appears on the each single frame prediction
            labels = np.unique(gt_seg_list[f])
            labels = labels[labels != 0]  # Remove background class
            for obj in labels:
                class_lables[int(obj) - 1] = obj

            sfp_lookup = utils.to_torch(class_lables)

            # TODO: choose the range of classes for class conditioning
            sfp_interval = torch.ones_like(sfp_lookup) * -1
            cls_id = torch.unique(sfp_lookup)
            cls_id = cls_id[cls_id != -1]

            if len(cls_id) > 0:
                cls_list = torch.arange(cls_id.min(), cls_id.max() + 1)

                if cls_list.min() > 10:
                    min_cls = cls_list.min()
                    l1 = torch.arange(min_cls - 10, min_cls)
                    cls_list = torch.cat([l1, cls_list], dim=0)

                if cls_list.max() < 81:
                    max_cls = cls_list.max() + 1
                    l2 = torch.arange(max_cls, max_cls + 10)
                    cls_list = torch.cat([cls_list, l2], dim=0)

                for obj in cls_list:
                    sfp_interval[int(obj) - 1] = obj

                lookup_list.append(sfp_interval)
            else:
                del_idx_list.append(f)

        print(self.num_frames[_video_name], del_idx_list)
        for idx in del_idx_list[::-1]:
            del image_list[idx]
            del info["frames"][idx]
            del info["single_frame_path"][idx]
            del info["crop_regions"][idx]
        info["num_frames"] = info["num_frames"] - len(del_idx_list)

        lookup_list = torch.stack(lookup_list, dim=0)  # T*CK:91
        selector_list = torch.ones_like(lookup_list)  # T*CK:91
        selector_list[lookup_list == -1] = 0

        # (num_frames, 3, 720, 1280)
        image_list = torch.stack(image_list, dim=0)
        # homo_mat_list = np.stack(homo_mat_list, axis=0)
        # (K:num_objects, T:num_frames, C:1, H:180, W:320)
        # target_dilated_hm_list = target_dilated_hm_list.unsqueeze(2)

        data = {}
        data["rgb"] = image_list
        # data["target_dilated_hm"] = target_dilated_hm_list
        # data["cls_gt"] = cls_gt
        # data["gt_homo"] = homo_mat_list
        data["selector"] = selector_list
        data["lookup"] = lookup_list
        data["info"] = info

        print(
            len(image_list),
            len(selector_list),
            len(lookup_list),
            info["num_frames"],
            len(info["frames"]),
            len(info["single_frame_path"]),
        )

        return data
