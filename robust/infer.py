"""
testing file for Nie et al. (A robust and efficient framework for sports-field registration)
"""

import os
import os.path as osp
import sys

sys.path.append("..")
import time
from glob import glob

import cv2
import matplotlib.pyplot as plt
import metrics
import numpy as np
import skimage.segmentation as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from options import CustomOptions
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import EncDec
from pitch_dataset import PitchDataset


# Get input arguments
opt = CustomOptions(train=False)
opt = opt.parse()

# Setup GPU
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
print("CUDA_VISIBLE_DEVICES: %s" % opt.gpu_ids)
device = torch.device("cuda:0")
print("device: %s" % device)


def postprocessing(scores, pred, target, num_classes, nms_thres):

    # TODO: decode the heatmaps into keypoint sets using non-maximum suppression
    pred_cls_dict = {k: [] for k in range(1, num_classes)}

    for cls in range(1, num_classes):
        pred_inds = pred == cls

        # implies the current class does not appear in this heatmaps
        if not np.any(pred_inds):
            continue

        values = scores[pred_inds]
        max_score = values.max()
        max_index = values.argmax()

        indices = np.where(pred_inds)
        coords = list(zip(indices[0], indices[1]))

        # the only keypoint with max confidence is greater than threshold or not
        if max_score >= nms_thres:
            pred_cls_dict[cls].append(max_score)
            pred_cls_dict[cls].append(coords[max_index])

    gt_cls_dict = {k: [] for k in range(1, num_classes)}
    for cls in range(1, num_classes):
        gt_inds = target == cls

        # implies the current class does not appear in this heatmaps
        if not np.any(gt_inds):
            continue
        coords = np.argwhere(gt_inds)[0]

        # coordinate order is (y, x)
        gt_cls_dict[cls].append((coords[0], coords[1]))

    return gt_cls_dict, pred_cls_dict


def infer_postprocessing(scores, pred, num_classes, nms_thres):

    # TODO: decode the heatmaps into keypoint sets using non-maximum suppression
    pred_cls_dict = {k: [] for k in range(1, num_classes)}

    for cls in range(1, num_classes):
        pred_inds = pred == cls

        # implies the current class does not appear in this heatmaps
        if not np.any(pred_inds):
            continue

        values = scores[pred_inds]
        max_score = values.max()
        max_index = values.argmax()

        indices = np.where(pred_inds)
        coords = list(zip(indices[0], indices[1]))

        # the only keypoint with max confidence is greater than threshold or not
        if max_score >= nms_thres:
            pred_cls_dict[cls].append(max_score)
            pred_cls_dict[cls].append(coords[max_index])

    return pred_cls_dict


def class_mapping(rgb):

    # TODO: class mapping
    template = utils.gen_template_grid()  # grid shape (91, 3), (x, y, label)
    src_pts = rgb.copy()
    cls_map_pts = []

    for ind, elem in enumerate(src_pts):
        coords = np.where(elem[2] == template[:, 2])[0]  # find correspondence
        cls_map_pts.append(template[coords[0]])
    dst_pts = np.array(cls_map_pts, dtype=np.float32)

    return src_pts[:, :2], dst_pts[:, :2]


def infer():
    num_classes = 92
    non_local = bool(opt.use_non_local)
    layers = 18

    # Initialize models
    model = EncDec(layers, num_classes, non_local).to(device)

    print("Loading data...")
    test_dataset = PitchDataset(opt.custom_data_root)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # Set data path
    denorm = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    test_visual_dir = osp.join(exp_name_path, "infer_visual")
    os.makedirs(test_visual_dir, exist_ok=True)

    iou_visual_dir = osp.join(test_visual_dir, "iou")
    os.makedirs(iou_visual_dir, exist_ok=True)

    homo_visual_dir = osp.join(exp_name_path, "homography")
    # homo_visual_dir = osp.join(test_visual_dir, "homography")
    os.makedirs(homo_visual_dir, exist_ok=True)

    # field_model = Image.open(osp.join(opt.template_path, "worldcup_field_model.png"))

    # TODO:: Load pretrained model or resume training
    if len(opt.ckpt_path) > 0:
        load_weights_path = opt.ckpt_path
        print("Loading weights: ", load_weights_path)
        assert osp.isfile(load_weights_path), "Error: no checkpoints found"
        checkpoint = torch.load(load_weights_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        print("Checkpoint Epoch: ", epoch)

    sfp_out_path = "SingleFramePredict_finetuned_with_normalized"

    print("Testing...")
    model.eval()

    test_progress_bar = tqdm(
        enumerate(test_loader), total=len(test_loader), leave=False
    )
    # test_progress_bar.set_description(f"Epoch: {epoch}/{opt.train_epochs}")

    with torch.no_grad():
        for step, (image, filename) in test_progress_bar:
            image = image.to(device)
            filename_head = osp.splitext(osp.basename(filename[0]))[0]
            vid_name = osp.basename(osp.dirname(osp.dirname(filename[0])))

            pred_heatmap = model(image)

            pred_heatmap = torch.softmax(pred_heatmap, dim=1)
            scores, pred_heatmap = torch.max(pred_heatmap, dim=1)
            scores = scores[0].detach().cpu().numpy()
            pred_heatmap = pred_heatmap[0].detach().cpu().numpy()
            # target = target[0].cpu().numpy()

            pred_cls_dict = infer_postprocessing(
                scores, pred_heatmap, num_classes, opt.nms_thres
            )

            image = utils.im_to_numpy(denorm(image[0]))

            # TODO: show keypoints visual result after postprocessing
            pred_keypoints = np.zeros_like(pred_heatmap, dtype=np.uint8)
            pred_rgb = []
            for ind, (pk, pv) in enumerate(pred_cls_dict.items()):
                if pv:
                    pred_keypoints[pv[1][0], pv[1][1]] = pk  # (H, W)
                    # camera view point sets (x, y, label) in rgb domain not heatmap domain
                    pred_rgb.append([pv[1][1] * 4, pv[1][0] * 4, pk])
            pred_rgb = np.asarray(pred_rgb, dtype=np.float32)  # (?, 3)
            pred_homo = None
            if pred_rgb.shape[0] >= 4:  # at least four points
                src_pts, dst_pts = class_mapping(pred_rgb)
                pred_homo, _ = cv2.findHomography(
                    src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 10
                )

            # # TODO: save pic
            if True:
                # if False:
                pred_keypoints = ss.expand_labels(pred_keypoints, distance=5)

                # TODO: save undilated heatmap for each testing video
                vid_path = osp.join(exp_name_path, vid_name, sfp_out_path)
                os.makedirs(vid_path, exist_ok=True)
                cv2.imwrite(
                    osp.join(vid_path, "{}.png".format(filename_head)), pred_keypoints
                )

                # plt.imsave(osp.join(test_visual_dir,
                #            'test_%05d_%05d_rgb.jpg' % (epoch, step)), image)
                test_visual_path = osp.join(test_visual_dir, vid_name)
                os.makedirs(test_visual_path, exist_ok=True)
                plt.imsave(
                    osp.join(
                        test_visual_path,
                        "test_{:05d}_{}_pred.png".format(epoch, filename_head),
                    ),
                    pred_heatmap,
                    vmin=0,
                    vmax=91,
                )

                # pred_keypoints = ss.expand_labels(
                #     pred_keypoints, distance=5)
                plt.imsave(
                    osp.join(
                        test_visual_path,
                        "test_{:05d}_{}_pred_keypts.png".format(epoch, filename_head),
                    ),
                    pred_keypoints,
                    vmin=0,
                    vmax=91,
                )

            if True:
                # if False:
                if pred_rgb.shape[0] >= 4 and pred_homo is not None:
                    homo_vid_path = osp.join(
                        homo_visual_dir,
                        vid_name,
                    )
                    os.makedirs(homo_vid_path, exist_ok=True)
                    np.save(
                        osp.join(
                            homo_vid_path,
                            "test_{:05d}_{}_pred_homography.npy".format(
                                epoch, filename_head
                            ),
                        ),
                        pred_homo,
                    )


def main():
    infer()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Done...Take {(time.time() - start_time):.4f} (sec)")
