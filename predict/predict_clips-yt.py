import argparse
import os
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import h5py
import time
import datetime
import json

from hand_and_face_crop import video_holistic
import predict_dino

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_batch_idx(num_samples, batch_size):
    steps = int(np.floor(num_samples / batch_size))
    idx = []
    for i in range(1, steps + 1):
        start = (i - 1) * batch_size
        end = i * batch_size
        idx.append((start, end))

    if steps * batch_size != num_samples:
        idx.append((steps * batch_size, num_samples))
    return idx


def save_to_h5(features_list_h5, label, index_dataset, chunk_batch, chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        features_list_h5.resize(chunk_batch * chunk_size, axis=0)
    features_list_h5[index_dataset:index_dataset + chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch


def add_to_h5(clip_name, clip_features, index_dataset, chunk_batch, chunk_size):
    feature_shape = clip_features.shape
    features_list_h5 = video_h5.create_dataset(
        clip_name,
        shape=feature_shape,
        maxshape=(None, feature_shape[-1]),
        dtype=np.dtype('float16')
    )
    num_full_chunks = len(clip_features) // chunk_size
    last_chunk_size = len(clip_features) % chunk_size
    for c in range(num_full_chunks):
        feature = clip_features[index_dataset:index_dataset + chunk_size]
        index_dataset, chunk_batch = save_to_h5(features_list_h5, feature, index_dataset, chunk_batch,
                                                chunk_size)
    if last_chunk_size > 0:
        feature = clip_features[index_dataset:index_dataset + last_chunk_size]
        index_dataset, chunk_batch = save_to_h5(features_list_h5, feature, index_dataset, chunk_batch,
                                                last_chunk_size)


def video_to_embeddings(images, model, batch_size=128):
    video_features = []
    total_frames = len(images)
    for idx in range(0, total_frames, batch_size):
        idxs = list(range(idx, min(idx + batch_size, total_frames)))
        batch = [images[idx] for idx in idxs]

        with torch.no_grad():
            features = predict_dino.dino_predict(batch, model, predict_dino.transform_dino, device)
            video_features.append(features)
    video_features = np.concatenate(video_features, axis=0)
    return video_features


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--clip_folder', type=str)
    parser.add_argument('--face_checkpoint', default="", type=str)
    parser.add_argument('--hand_checkpoint', default="", type=str)
    parser.add_argument('--num_splits', type=int)
    parser.add_argument('--split', type=int)
    parser.add_argument('--split_name', default="train", type=str)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    output_file_name = f'YouTubeASL.dino.{args.split_name}.{args.split}.h5'
    meta_file_name = f"YouTubeASL.dino.{args.split_name}.{args.split}.json"
    os.makedirs(args.output_folder, exist_ok=True)

    # load clip paths
    clip_names = os.listdir(args.clip_folder)
    clip_names = [file for file in clip_names if ".mp4" in file]

    # group clips based on the video name
    clip_names_sorted = np.sort(clip_names)
    video_to_clips = {}
    for file in clip_names_sorted:
        file = os.path.join(args.clip_folder, file)
        name = os.path.basename(file)
        name_split = name.split(".")[:-1]
        video_name = ".".join(name_split[:-1])
        if video_name in video_to_clips:
            video_to_clips[video_name].append(file)
        else:
            video_to_clips[video_name] = [file]

    # split to chunks
    num_samples = len(video_to_clips)
    batch_size = int(np.ceil(num_samples / (args.num_splits)))
    idxs = get_batch_idx(num_samples, batch_size)
    start, end = idxs[args.split]
    video_names = list(video_to_clips.keys())
    video_names.sort()
    video_names = video_names[start:end]
    print(idxs)
    print(f"Number of splits: {len(idxs)}")
    print(f"Number of videos: {len(video_names)}")

    # load models
    face_model = predict_dino.create_dino_model(args.face_checkpoint)
    hand_model = predict_dino.create_dino_model(args.hand_checkpoint)
    face_model.to(device)
    hand_model.to(device)

    # h5py file initialization
    f_out = h5py.File(os.path.join(args.output_folder, output_file_name), 'w')

    # predict
    prediction_times = []
    frames = []
    metadata = {}
    for video_idx, video_name in enumerate(video_names):
        clip_paths = video_to_clips[video_name]
        metadata[video_name] = args.split
        video_h5 = f_out.create_group(video_name)
        start_time = time.time()
        for clip_path in clip_paths:
            # parse name
            name = os.path.basename(clip_path)
            name_split = name.split(".")[:-1]
            clip_name = ".".join(name_split)

            # predict video features
            face, hand_left, hand_right = video_holistic(clip_name, os.path.dirname(clip_path))
            face_features = video_to_embeddings(face, face_model, batch_size=32)
            left_features = video_to_embeddings(hand_left, hand_model, batch_size=32)
            right_features = video_to_embeddings(hand_right, hand_model, batch_size=32)

            features = np.concatenate([face_features, left_features, right_features], 1)

            # save features in hd5
            add_to_h5(
                clip_name,
                features,
                index_dataset=0,
                chunk_batch=1,
                chunk_size=len(features)
            )

            frames.append(len(face))
        # print stats
        end_time = time.time()
        prediction_times.append(end_time - start_time)

        print(f"[{video_idx + 1}/{len(video_names)}]")
        print(f"average time: {np.mean(prediction_times):.3f}")
        print(f"average frames: {np.mean(frames):.2f}")
        secs = ((len(video_names) - (video_idx + 1)) * np.mean(prediction_times))
        print(f"eta: {str(datetime.timedelta(seconds=secs)).split('.')[0]}")
        print()

    with open(os.path.join(args.output_folder, meta_file_name), "w") as f:
        json.dump(metadata, f)

    f_out.close()

    # merge json files
    json_files = [name for name in os.listdir(args.output_folder) if ".json" in name]
    if args.num_splits == len(json_files):
        merged_data = {}
        for name in json_files:
            path = os.path.join(args.output_folder, name)
            with open(path, "r") as f:
                data = json.load(f)
            merged_data.update(data)

        with open(os.path.join(args.output_folder, f"YouTubeASL.dino.{args.split_name}.json"), "w") as f:
            json.dump(merged_data, f)
