import itertools
import os
import os.path as osp
import shutil
import time
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.lib import tile
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import metrics
import utils
from models.eval_network import EvalKpSFR
from models.inference_core import InferenceCore
from options import CustomOptions
from pitch_dataset import PitchDataset

# Get input arguments
opt = CustomOptions(train=False)
opt = opt.parse()

# Log on tensorboard
# writer = SummaryWriter('runs/' + opt.name)

# Setup GPU
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
print("CUDA Visible Devices: %s" % opt.gpu_ids)
device = torch.device("cuda:0")
print("device: %s" % device)


def calc_euclidean_distance(a, b, _norm=np.linalg.norm, axis=None):
    return _norm(a - b, axis=axis)


def my_mseloss(gt, pred):
    return torch.mean(torch.square(pred - gt))


def postprocessing(scores, pred, num_classes, nms_thres):

    # TODO: decode the heatmaps into keypoint sets using non-maximum suppression
    pred_cls_dict = {k: [] for k in range(1, num_classes)}
    for cls in range(1, num_classes):
        pred_inds = pred == cls

        # implies the current class does not appear in this heatmaps
        if not np.any(pred_inds):
            continue

        values = scores[pred_inds]
        max_score = values.max()

        val_inds = np.where(values == max_score)[0]

        indices = np.where(pred_inds)
        coords = list(zip(indices[0], indices[1]))

        l = []
        for idx in range(val_inds.shape[0]):
            l.append(coords[val_inds[idx]])
        l = np.array(l).mean(axis=0).astype(np.int64)

        # the only keypoint with max confidence is greater than threshold or not
        if max_score >= nms_thres:
            pred_cls_dict[cls].append(max_score)
            pred_cls_dict[cls].append(l)

    return pred_cls_dict


def calc_keypts_metrics(gt_cls_dict, pred_cls_dict, pr_thres):

    num_gt_pos = 0
    num_pred_pos = 0
    num_both_keypts_appear = 0
    tp = 0
    mse_loss = 0.0

    for (gk, gv), (pk, pv) in zip(gt_cls_dict.items(), pred_cls_dict.items()):
        if gv:
            num_gt_pos += 1

        if pv:
            num_pred_pos += 1

        if gv and pv:
            num_both_keypts_appear += 1
            mse_loss += my_mseloss(torch.FloatTensor(gv[0]), torch.FloatTensor(pv[1]))

            if calc_euclidean_distance(np.array(gv[0]), np.array(pv[1])) <= pr_thres:
                tp += 1

    if num_both_keypts_appear == 0:
        return 0.0, 0.0, 0.0
    return tp / num_pred_pos, tp / num_gt_pos, mse_loss / num_both_keypts_appear


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


def test():
    num_classes = 92
    # num_objects = opt.num_objects
    num_objects = 91
    non_local = bool(opt.use_non_local)
    model_archi = opt.model_archi

    # Initialize models
    eval_model = EvalKpSFR(
        model_archi=model_archi, num_objects=num_objects, non_local=non_local
    ).to(device)

    print("Loading time sequence worldcup testing data...")
    test_dataset = PitchDataset(
        root=opt.custom_data_root,
        num_objects=num_objects,
        sfp_finetuned=opt.sfp_finetuned,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    total_epoch = opt.train_epochs

    # Set data path
    denorm = utils.UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    exp_name_path = osp.join(opt.checkpoints_dir, opt.name)
    # test_visual_dir = osp.join(exp_name_path, "imgs", "infer_visual")
    # if osp.exists(test_visual_dir):
    #     print(f"Remove directory: {test_visual_dir}")
    #     shutil.rmtree(test_visual_dir)
    # print(f"Create directory: {test_visual_dir}")
    # os.makedirs(test_visual_dir, exist_ok=True)

    # iou_visual_dir = osp.join(test_visual_dir, "iou")
    # os.makedirs(iou_visual_dir, exist_ok=True)

    # homo_visual_dir = osp.join(exp_name_path, "homography")
    # os.makedirs(homo_visual_dir, exist_ok=True)

    # field_model = Image.open(osp.join(opt.template_path, "worldcup_field_model.png"))

    # TODO: Load pretrained model or resume training
    if len(opt.ckpt_path) > 0:
        load_weights_path = opt.ckpt_path
        print("Loading weights: ", load_weights_path)
        assert osp.isfile(load_weights_path), "Error: no checkpoints found"
        checkpoint = torch.load(load_weights_path, map_location=device)
        eval_model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        print("Checkpoint Epoch:", epoch)

    print("Testing...")
    eval_model.eval()
    test_progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    test_progress_bar.set_description(f"Epoch: {epoch}/{total_epoch}")

    total_process_time = 0
    total_frames = 0

    template = utils.gen_template_grid()
    total_pred_kps = defaultdict(list)
    total_temp_kps = defaultdict(list)
    with torch.no_grad():
        for step, data in test_progress_bar:
            image = data["rgb"].to(device)  # b*t*c*h*w
            # target_dilated_hm = data["target_dilated_hm"][0].to(device)  # k*t*1*h*w
            lookup = data["lookup"][0].to(device)  # k:91 or t*k
            info = data["info"]
            k = info["num_objects"]
            # sfp_path = info["single_frame_path"][0]
            vid_name = info["name"][0]

            torch.cuda.synchronize()
            process_begin = time.time()

            processor = InferenceCore(eval_model, image, device, k, lookup)
            # selector does not use
            processor.interact(0, image.shape[1])

            # size = target_dilated_hm.shape[-2:]
            size = test_dataset.frame_h // 4, test_dataset.frame_w // 4
            out_masks = torch.zeros((processor.t, 1, *size), device=device)
            out_scores = torch.zeros_like(out_masks)

            for ti in range(processor.t):
                prob = processor.prob[:, ti]
                out_scores[ti], out_masks[ti] = torch.max(prob, dim=0)  # 1*h*w
            out_masks = out_masks.detach().cpu().numpy()[:, 0]  # t*h*w
            out_scores = out_scores.detach().cpu().numpy()[:, 0]  # t*h*w

            image = np.transpose(denorm(image[0]).detach().cpu().numpy(), (0, 2, 3, 1))  # t*h*w*c

            torch.cuda.synchronize()
            total_process_time += time.time() - process_begin
            total_frames += out_masks.shape[0]

            if opt.train_stage == 0 and opt.target_image:
                tmp_step = step
                step = int(opt.target_image.pop())

            for ti in range(processor.t):
                x_offset, y_offset = (
                    info["crop_regions"][ti][0].item(),
                    info["crop_regions"][ti][1].item(),
                )
                print(f"Current frame is {ti}")

                print("scores: ", out_scores[ti].min(), out_scores[ti].max())
                pred_cls_dict = postprocessing(
                    out_scores[ti],
                    out_masks[ti],
                    num_classes,
                    opt.nms_thres,
                )
                # No any point after postprocessing
                if not any(pred_cls_dict.values()):
                    print(f"not keypts at {ti}")
                    plt.imsave(
                        osp.join(
                            exp_name_path,
                            "imgs",
                            "test_%05d_%05d_pred_not_keypts.png" % (epoch, step),
                        ),
                        out_masks[ti],
                        vmin=0,
                        vmax=processor.k,
                    )
                    continue

                # TODO: show keypoints visual result after postprocessing
                pred_keypoints = np.zeros_like(out_masks[0])
                pred_rgb = []
                for ind, (pk, pv) in enumerate(pred_cls_dict.items()):
                    if pv:
                        pred_keypoints[pv[1][0], pv[1][1]] = pk  # (H, W)
                        # camera view point sets (x, y, label) in rgb domain not heatmap domain
                        x, y = pv[1][1] * 4, pv[1][0] * 4
                        temp_x, temp_y = template[pk - 1][:2]
                        pred_rgb.append([x, y, pk])
                        total_pred_kps[pk].append([x + x_offset, y + y_offset])
                        total_temp_kps[pk].append([temp_x, temp_y])
                pred_rgb = np.asarray(pred_rgb, dtype=np.float32)  # (?, 3)

                pred_homo = None
                if pred_rgb.shape[0] >= 4:  # at least four points
                    src_pts, dst_pts = class_mapping(pred_rgb)

                    pred_homo, _ = cv2.findHomography(
                        src_pts.reshape(-1, 1, 2),
                        dst_pts.reshape(-1, 1, 2),
                        cv2.RANSAC,
                        10,
                    )

                # TODO: save undilated heatmap for each testing video
                # vid_path = osp.join(homo_visual_dir, "custom_data")
                # os.makedirs(vid_path, exist_ok=True)
                vid_path_m = osp.join(
                    exp_name_path, vid_name, model_archi, "custom_data"
                )  # for evaluate worldcup test set

                os.makedirs(vid_path_m, exist_ok=True)
                cv2.imwrite(osp.join(vid_path_m, "%05d.png" % ti), np.uint8(pred_keypoints))

                # TODO: save heatmap for visual result
                if False:
                    plt.imsave(
                        osp.join(
                            test_visual_dir,
                            "test_%05d_%05d_pred_seg%02d.png" % (epoch, step, ti),
                        ),
                        out_masks[ti],
                        vmin=0,
                        vmax=processor.k,
                    )
                    pred_keypoints = ss.expand_labels(pred_keypoints, distance=5)
                    plt.imsave(
                        osp.join(
                            test_visual_dir,
                            "test_%05d_%05d_pred_keypts%02d.png" % (epoch, step, ti),
                        ),
                        pred_keypoints,
                        vmin=0,
                        vmax=processor.k,
                    )

                # TODO: save homography
                # if False:
                if True:
                    if pred_rgb.shape[0] >= 4 and pred_homo is not None:
                        homo_vid_path = osp.join(exp_name_path, vid_name, "homography")
                        os.makedirs(homo_vid_path, exist_ok=True)

                        np.save(
                            osp.join(
                                homo_vid_path,
                                f"test_{epoch:05d}_{step:05d}_pred_homography{ti:02d}.npy",
                            ),
                            pred_homo,
                        )

            if opt.train_stage == 0 and opt.target_image:
                step = tmp_step

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

            del image
            # del target_dilated_hm
            del lookup
            del processor

        # writer.add_scalar(
        #     'Metrics/median Projection Error', median_proj_error, epoch)
        # writer.add_scalar(
        #     'Metrics/median Reprojection Error', median_reproj_error, epoch)

        with open(osp.join(exp_name_path, "%05d.txt" % epoch), "w") as out_file:
            out_file.write(f"Loading weights: {load_weights_path}")
            out_file.write("\n")
            # out_file.write(f"Path of single frame prediction: {sfp_path}")
            # out_file.write("\n")
            out_file.write(f"Model architecture: {model_archi}")
            out_file.write("\n")

        print("Total processing time: ", total_process_time)
        print("Total processed frames: ", total_frames)
        # print(f"FPS: {(total_frames / total_process_time):.3f}")


def main():
    test()
    # writer.flush()
    # writer.close()


if __name__ == "__main__":

    start_time = time.time()
    main()
    print(f"Done...Take {(time.time() - start_time):.4f} (sec)")
