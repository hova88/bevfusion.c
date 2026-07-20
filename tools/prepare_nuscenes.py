#!/usr/bin/env python3
"""Prepare a deterministic OpenPCDet BEVFusion BFI from nuScenes."""

import argparse
import os
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from pyquaternion import Quaternion

from pack_frame import FIELDS, canonical, write_bfi

CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def sensor_to_global(nusc: NuScenes, sample_data: dict) -> np.ndarray:
    calibrated = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    return (transform_matrix(pose["translation"], Quaternion(pose["rotation"])) @
            transform_matrix(calibrated["translation"], Quaternion(calibrated["rotation"])))


def load_points(nusc: NuScenes, sample: dict, sweeps: int, seed: int) -> np.ndarray:
    reference = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    reference_from_global = np.linalg.inv(sensor_to_global(nusc, reference))
    records, current = [], reference
    while len(records) < sweeps - 1:
        if current["prev"]:
            current = nusc.get("sample_data", current["prev"])
            records.append(current)
        elif records:
            records.append(records[-1])
        else:
            records.append(reference)
    rng = np.random.RandomState(seed)
    order = rng.choice(len(records), sweeps - 1, replace=False) if records else []
    current_points = np.fromfile(nusc.get_sample_data_path(reference["token"]),
                                 np.float32).reshape(-1, 5)[:, :4]
    parts = [np.c_[current_points, np.zeros((len(current_points), 1), np.float32)]]
    reference_time = reference["timestamp"] * 1e-6
    for index in order:
        record = records[index]
        points = np.fromfile(nusc.get_sample_data_path(record["token"]),
                             np.float32).reshape(-1, 5)[:, :4]
        points = points[~((np.abs(points[:, 0]) < 1.0) & (np.abs(points[:, 1]) < 1.0))]
        transform = reference_from_global @ sensor_to_global(nusc, record)
        xyz1 = np.c_[points[:, :3], np.ones(len(points), np.float32)]
        xyz = (transform @ xyz1.T).T[:, :3]
        lag = np.full((len(points), 1), reference_time - record["timestamp"] * 1e-6,
                      np.float32)
        parts.append(np.c_[xyz, points[:, 3], lag])
    result = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    rng.shuffle(result)
    return result


def load_cameras(nusc: NuScenes, sample: dict) -> tuple[np.ndarray, ...]:
    reference = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    reference_from_global = np.linalg.inv(sensor_to_global(nusc, reference))
    images, intrinsics, camera_to_lidar, lidar_to_image = [], [], [], []
    image_augmentation = []
    for channel in CAMERAS:
        record = nusc.get("sample_data", sample["data"][channel])
        calibrated = nusc.get("calibrated_sensor", record["calibrated_sensor_token"])
        c2l = reference_from_global @ sensor_to_global(nusc, record)
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = np.asarray(calibrated["camera_intrinsic"], np.float32)
        camera_to_lidar.append(c2l.astype(np.float32))
        intrinsics.append(intrinsic)
        lidar_to_image.append(intrinsic @ np.linalg.inv(c2l).astype(np.float32))
        with Image.open(nusc.get_sample_data_path(record["token"])) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            resized = (int(width * 0.48), int(height * 0.48))
            crop_x = int(max(0, resized[0] - 704) / 2)
            crop_y = resized[1] - 256
            processed = rgb.resize(resized).crop((crop_x, crop_y, crop_x + 704,
                                                  crop_y + 256))
            pixels = np.asarray(processed, dtype=np.float32) / np.float32(255.0)
        pixels = ((pixels - np.array([0.485, 0.456, 0.406], np.float32)) /
                  np.array([0.229, 0.224, 0.225], np.float32))
        images.append(pixels.transpose(2, 0, 1))
        augmentation = np.eye(4, dtype=np.float32)
        augmentation[0, 0] = augmentation[1, 1] = 0.48
        augmentation[0, 3], augmentation[1, 3] = -crop_x, -crop_y
        image_augmentation.append(augmentation)
    return tuple(np.ascontiguousarray(value, dtype=np.float32) for value in
                 (images, intrinsics, camera_to_lidar, image_augmentation,
                  lidar_to_image))


def main() -> None:
    default_root = Path(os.environ.get(
        "NUSCENES_ROOT", Path.home() / "datasets" / "nuscenes"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_bfi", type=Path)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--token")
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                        help="pins OpenPCDet sweep and point shuffles")
    args = parser.parse_args()
    if args.sweeps < 1:
        raise ValueError("sweeps must be positive")
    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=False)
    if args.token:
        sample = nusc.get("sample", args.token)
    else:
        sample = nusc.sample[args.index]
    points = load_points(nusc, sample, args.sweeps, args.seed)
    images, intrinsics, c2l, image_aug, l2i = load_cameras(nusc, sample)
    raw = (images, points, intrinsics, c2l, image_aug, np.eye(4, dtype=np.float32), l2i)
    arrays = [canonical(name, value, shape) for (name, shape), value in zip(FIELDS, raw)]
    write_bfi(arrays, args.output_bfi)
    print(f"sample_token={sample['token']} seed={args.seed} sweeps={args.sweeps}")


if __name__ == "__main__":
    main()
