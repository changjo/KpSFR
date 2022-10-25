import argparse
import os
import os.path as osp

import cv2
import numpy as np


ORIGINAL_WIDTH = 1280
ORIGINAL_HEIGHT = 720


def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


def birds_eye_view(img, H, width, height):
    return cv2.warpPerspective(img, H, (width, height))


def run(args):
    H = np.load(args.homography)

    S = np.eye(3)
    S[0, 0] = S[1, 1] = args.out_scale

    img = cv2.imread(args.image)
    h_img, w_img = img.shape[:2]
    if args.crop is not None:
        left, top, right, bottom = args.crop

        w_ratio = ORIGINAL_WIDTH / (right - left)
        h_ratio = ORIGINAL_HEIGHT / (bottom - top)

        left, right = left * w_ratio, right * w_ratio
        top, bottom = top * h_ratio, bottom * h_ratio

        # img = img[top:bottom, left:right]
        T_1 = np.eye(3)
        T_1[0, 2] = -left
        T_1[1, 2] = -top
        H = H @ T_1

    else:
        w_ratio = ORIGINAL_WIDTH / w_img
        h_ratio = ORIGINAL_HEIGHT / h_img

    new_w_img = w_img * w_ratio
    new_h_img = h_img * h_ratio

    points = np.array(
        [[0.0, 0.0], [new_w_img, 0.0], [new_w_img, new_h_img], [0.0, new_h_img]]
    ).reshape(-1, 1, 2)
    homo_points = cv2.perspectiveTransform(points, H).reshape(-1, 2)

    T_2 = np.eye(3)
    T_2[0, 2] = -min(homo_points[:, 0])
    T_2[1, 2] = -min(homo_points[:, 1])
    H = T_2 @ H

    H = S @ H

    img_resized = cv2.resize(img, (int(new_w_img), int(new_h_img)))
    img_bev = birds_eye_view(img_resized, H, args.width, args.height)
    out_dir = osp.join(osp.dirname(osp.dirname(args.image)), "bev")
    make_dir(out_dir)
    out_img_filename = osp.join(
        out_dir, osp.splitext(osp.basename(args.image))[0] + "_bev.jpg"
    )
    cv2.imwrite(out_img_filename, img_bev)
    print(f"Saved to {out_img_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--homography", type=str, required=True, help="Path to homography matrix"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--width", type=int, required=True, help="Width of birds eye view"
    )
    parser.add_argument(
        "--height", type=int, required=True, help="Height of birds eye view"
    )
    parser.add_argument(
        "--out_scale", type=int, default=20, help="Scale of output image"
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs="+",
        default=None,
        help="Crop region (left, top, right, bottom)",
    )
    args = parser.parse_args()

    run(args)
