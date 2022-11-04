"""
testing file for Nie et al. (A robust and efficient framework for sports-field registration)
"""

import itertools
import os
import os.path as osp
import sys

sys.path.append("..")

import random
import time
from collections import defaultdict
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as ss
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import utils
from models.model import EncDec
from options import CustomOptions
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


def get_all_comb(total_pred_kps, total_temp_kps, shuffle=False):
    pk_list = list(total_pred_kps.keys())
    pks_comb = list(itertools.combinations(pk_list, r=4))
    length = {pk: len(total_pred_kps[pk]) for pk in total_pred_kps}

    all_comb_idx = defaultdict(list)
    all_comb_pred = []
    all_comb_temp = []
    for pks in pks_comb:
        p_idx_list = [list(range(length[pk])) for pk in pks]
        sub_comb_idx = np.array(np.meshgrid(*p_idx_list)).T.reshape(-1, 4)
        all_comb_idx[pks].append(sub_comb_idx)
        for indices in sub_comb_idx:
            four_points_pred = [total_pred_kps[pks[i]][indices[i]] for i in range(4)]
            four_points_temp = [total_temp_kps[pks[i]][0] for i in range(4)]
            all_comb_pred.append(four_points_pred)
            all_comb_temp.append(four_points_temp)

    if shuffle:
        np.random.seed(8888)
        rand_idx_list = np.random.permutation(len(all_comb_pred))
        all_comb_pred = [all_comb_pred[i] for i in rand_idx_list]
        all_comb_temp = [all_comb_temp[i] for i in rand_idx_list]

    return all_comb_pred, all_comb_temp


def find_min_cost_homography(all_comb_pred, all_comb_temp, total_pred_pts, total_temp_pts):
    total_pred_pts = np.float32(total_pred_pts).reshape(-1, 2)
    total_temp_pts = np.float32(total_temp_pts).reshape(-1, 2)

    best_H = None
    min_cost = 100000
    for i in trange(len(all_comb_pred)):
        pred_pts = np.array(all_comb_pred[i], dtype=np.float32).reshape(-1, 1, 2)
        temp_pts = np.array(all_comb_temp[i], dtype=np.float32).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(pred_pts, temp_pts)

        if H is not None:
            homo_total_pred_pts = cv2.perspectiveTransform(
                total_pred_pts.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            cost = np.linalg.norm(total_temp_pts - homo_total_pred_pts, axis=1).mean()
            if cost < min_cost:
                min_cost = cost
                best_H = H

    return best_H


def write_txt(data, filename):
    with open(filename, "w") as f:
        for i in range(len(data)):
            f.write(str(data[i]) + "\n")


def infer():
    num_classes = 92
    non_local = bool(opt.use_non_local)
    layers = 18

    # Initialize models
    model = EncDec(layers, num_classes, non_local).to(device)

    print("Loading data...")
    test_dataset = PitchDataset(opt.custom_data_root, crop_ratio=0.6)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # Set data path
    denorm = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    # test_visual_dir = osp.join(exp_name_path, "infer_visual")
    # os.makedirs(test_visual_dir, exist_ok=True)

    # iou_visual_dir = osp.join(test_visual_dir, "iou")
    # os.makedirs(iou_visual_dir, exist_ok=True)

    # homo_visual_dir = osp.join(exp_name_path, "homography")
    # # homo_visual_dir = osp.join(test_visual_dir, "homography")
    # os.makedirs(homo_visual_dir, exist_ok=True)

    # crop_visual_dir = osp.join(exp_name_path, "crop")
    # os.makedirs(crop_visual_dir, exist_ok=True)

    # draw_visual_dir = osp.join(exp_name_path, "draw")
    # os.makedirs(draw_visual_dir, exist_ok=True)

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

    test_progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    # test_progress_bar.set_description(f"Epoch: {epoch}/{opt.train_epochs}")

    crop_region = test_dataset.crop_region
    template = utils.gen_template_grid()
    with torch.no_grad():
        for step, (image_dict, filename) in test_progress_bar:
            total_pred_kps = defaultdict(list)
            total_temp_kps = defaultdict(list)
            filename_head = osp.splitext(osp.basename(filename[0]))[0]
            vid_name = osp.basename(osp.dirname(osp.dirname(filename[0])))
            for j, crop_loc in enumerate(test_dataset.crop_order):
                image = image_dict[crop_loc]
                image = image.to(device)
                pred_heatmap = model(image)
                pred_heatmap = torch.softmax(pred_heatmap, dim=1)
                scores, pred_heatmap = torch.max(pred_heatmap, dim=1)
                scores = scores[0].detach().cpu().numpy()
                pred_heatmap = pred_heatmap[0].detach().cpu().numpy()
                # target = target[0].cpu().numpy()

                pred_cls_dict = infer_postprocessing(
                    scores, pred_heatmap, num_classes, opt.nms_thres
                )

                x_offset, y_offset = (
                    crop_region[crop_loc][0],
                    crop_region[crop_loc][1],
                )

                image = utils.im_to_numpy(denorm(image[0]))
                image_copy = image.copy() * 255
                crop_visual_dir = osp.join(exp_name_path, vid_name, "crop")
                os.makedirs(crop_visual_dir, exist_ok=True)
                cv2.imwrite(
                    osp.join(crop_visual_dir, "{}_{}.jpg".format(filename_head, j)),
                    (image_copy)[:, :, ::-1],
                )

                crop_region_txt_filename = osp.join(
                    crop_visual_dir, "{}_{}.txt".format(filename_head, j)
                )
                crop_region_string = ",".join([str(v) for v in crop_region[crop_loc]])
                write_txt([crop_region_string], crop_region_txt_filename)

                # TODO: show keypoints visual result after postprocessing
                pred_keypoints = np.zeros_like(pred_heatmap, dtype=np.uint8)
                pred_rgb = []
                temp_kps = {}
                # print(pred_cls_dict)
                for ind, (pk, pv) in enumerate(pred_cls_dict.items()):
                    if pv:
                        pred_keypoints[pv[1][0], pv[1][1]] = pk  # (H, W)
                        # camera view point sets (x, y, label) in rgb domain not heatmap domain
                        x, y = pv[1][1] * 4, pv[1][0] * 4
                        temp_x, temp_y = template[pk - 1][:2]
                        pred_rgb.append([x, y, pk])
                        # new_x, new_y = pv[1][1] * 4 + x_offset, pv[1][0] * 4 + y_offset
                        total_pred_kps[pk].append([x + x_offset, y + y_offset])
                        total_temp_kps[pk].append([temp_x, temp_y])
                        temp_kps[pk] = (x, y)
                        cv2.circle(image_copy, (x, y), 3, (0, 0, 255), -1)
                        cv2.putText(image_copy, str(pk), (x, y), 0, 0.5, (0, 0, 255))

                pred_rgb = np.asarray(pred_rgb, dtype=np.float32)  # (?, 3)
                pred_homo = None
                inv_pred_homo = None
                # print(pred_rgb)
                if pred_rgb.shape[0] >= 4:  # at least four points
                    src_pts, dst_pts = class_mapping(pred_rgb)
                    pred_homo, _ = cv2.findHomography(
                        src_pts.reshape(-1, 1, 2),
                        dst_pts.reshape(-1, 1, 2),
                        cv2.RANSAC,
                        10,
                    )

                    # inv_pred_homo, _ = cv2.findHomography(
                    #     dst_pts.reshape(-1, 1, 2),
                    #     src_pts.reshape(-1, 1, 2),
                    #     cv2.RANSAC,
                    #     10,
                    # )

                if pred_homo is not None:
                    inv_pred_homo = np.linalg.inv(pred_homo)

                    for pk in range(1, len(template) + 1):  # temp_kps:
                        v = template[pk - 1][:2]
                        new_p = cv2.perspectiveTransform(
                            np.array(v).reshape(-1, 1, 2), inv_pred_homo
                        ).reshape(-1, 2)
                        t_x, t_y = int(new_p[0, 0]), int(new_p[0, 1])
                        cv2.circle(image_copy, (t_x, t_y), 3, (0, 255, 0), -1)
                        cv2.putText(image_copy, str(pk), (t_x, t_y), 0, 0.5, (0, 255, 0))

                draw_visual_dir = osp.join(exp_name_path, vid_name, "draw")
                os.makedirs(draw_visual_dir, exist_ok=True)
                cv2.imwrite(
                    osp.join(draw_visual_dir, "{}_{}.jpg".format(filename_head, j)),
                    (image_copy)[:, :, ::-1],
                )

                # Whiteboard plot
                margin = 100
                whiteboard = np.ones((1000, 1400, 3), dtype=np.uint8) * 255
                for pk in range(1, len(template) + 1):  # temp_kps:
                    v = template[pk - 1][:2] * 10
                    p = int(v[0]) + margin, int(v[1]) + margin
                    cv2.circle(whiteboard, p, 3, (0, 255, 0), -1)
                    cv2.putText(whiteboard, str(pk), p, 0, 0.5, (0, 255, 0))

                if pred_homo is not None:
                    for pk, v in temp_kps.items():
                        new_p = cv2.perspectiveTransform(
                            np.array(v, dtype=np.float32).reshape(-1, 1, 2), pred_homo
                        ).reshape(-1, 2)
                        new_p_x, new_p_y = (
                            int(new_p[0, 0] * 10) + margin,
                            int(new_p[0, 1] * 10) + margin,
                        )
                        cv2.circle(whiteboard, (new_p_x, new_p_y), 3, (255, 0, 0), -1)
                        cv2.putText(whiteboard, str(pk), (new_p_x, new_p_y), 0, 0.5, (255, 0, 0))

                cv2.imwrite(
                    osp.join(draw_visual_dir, "whiteboard_{}_{}.jpg".format(filename_head, j)),
                    (whiteboard)[:, :, ::-1],
                )

                # # TODO: save pic
                if True:
                    # if False:
                    pred_keypoints = ss.expand_labels(pred_keypoints, distance=5)

                    # TODO: save undilated heatmap for each testing video
                    vid_path = osp.join(exp_name_path, vid_name, sfp_out_path)
                    os.makedirs(vid_path, exist_ok=True)
                    cv2.imwrite(
                        osp.join(vid_path, "{}_{}.png".format(filename_head, j)),
                        pred_keypoints,
                    )

                    # plt.imsave(osp.join(test_visual_dir,
                    #            'test_%05d_%05d_rgb.jpg' % (epoch, step)), image)
                    test_visual_path = osp.join(exp_name_path, vid_name, "infer_visual")
                    os.makedirs(test_visual_path, exist_ok=True)
                    plt.imsave(
                        osp.join(
                            test_visual_path,
                            "test_{:05d}_{}_{}_pred.png".format(epoch, filename_head, j),
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
                            "test_{:05d}_{}_{}_pred_keypts.png".format(epoch, filename_head, j),
                        ),
                        pred_keypoints,
                        vmin=0,
                        vmax=91,
                    )

                if True:
                    # if False:
                    if pred_rgb.shape[0] >= 4 and pred_homo is not None:
                        homo_vid_path = osp.join(exp_name_path, vid_name, "homography")
                        os.makedirs(homo_vid_path, exist_ok=True)
                        np.save(
                            osp.join(
                                homo_vid_path,
                                "test_{:05d}_{}_{}_pred_homography.npy".format(
                                    epoch, filename_head, j
                                ),
                            ),
                            pred_homo,
                        )

            image_resized = image_dict["image_resized"]
            image_resized = utils.im_to_numpy(image_resized[0]) * 255
            image_resized = np.ascontiguousarray(image_resized)
            image_resized_copy = image_resized.copy()

            # template = utils.gen_template_grid()
            # print(template)
            # print(template.shape)

            # template_color = (255, 0, 0)
            for pk, v in total_pred_kps.items():
                for (x, y) in v:
                    rand_color = tuple(random.choices(range(256), k=3))
                    cv2.circle(image_resized, (x, y), 3, rand_color, -1)
                    cv2.putText(image_resized, str(pk), (x, y), 0, 0.5, rand_color)

                    # t_x, t_y, _ = template[pk - 1]

            cv2.imwrite(
                osp.join(draw_visual_dir, "{}.jpg".format(filename_head)),
                (image_resized)[:, :, ::-1],
            )

            print(total_pred_kps)
            print(total_temp_kps)

            all_comb_pred, all_comb_temp = get_all_comb(
                total_pred_kps, total_temp_kps, shuffle=True
            )
            arr_list = []
            for v in total_pred_kps.values():
                arr_list.append(np.array(v).reshape(-1, 2))
            total_pred_pts = np.concatenate(arr_list, axis=0)

            arr_list = []
            for v in total_temp_kps.values():
                arr_list.append(np.array(v).reshape(-1, 2))
            total_temp_pts = np.concatenate(arr_list, axis=0)

            best_H = find_min_cost_homography(
                all_comb_pred, all_comb_temp, total_pred_pts, total_temp_pts
            )

            if True:
                if best_H is not None:
                    homo_vid_path = osp.join(exp_name_path, vid_name, "homography")
                    os.makedirs(homo_vid_path, exist_ok=True)
                    np.save(
                        osp.join(
                            homo_vid_path,
                            "test_{:05d}_{}_pred_homography_total.npy".format(epoch, vid_name),
                        ),
                        best_H,
                    )

            S = np.eye(3)
            S[0, 0] = S[1, 1] = 20
            image_resized_homo = cv2.warpPerspective(
                image_resized_copy,
                S @ best_H,
                (3840, 2160),
            )
            cv2.imwrite(
                osp.join(draw_visual_dir, "{}_total_bev.jpg".format(filename_head)),
                (image_resized_homo)[:, :, ::-1],
            )


def main():
    infer()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Done...Take {(time.time() - start_time):.4f} (sec)")
