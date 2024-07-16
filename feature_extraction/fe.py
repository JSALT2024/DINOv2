import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import decord
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
import os
import pickle
import gzip
from pathlib import Path
import argparse
import json
import csv
import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#select one
DINO_PATH_FINETUNED_DOWNLOADED='/DinoFace/Checkpoint/teacher_checkpoint.pth'
#DINO_PATH_FINETUNED_DOWNLOADED='/DinoHand/Checkpoint/teacher_checkpoint.pth'




def get_mp4_files(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')

    # Use glob to find all .mp4 files
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))

    # Convert to absolute paths
    absolute_paths = [os.path.abspath(file) for file in mp4_files]

    return absolute_paths


def load_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


def is_string_in_file(file_path, target_string):
    try:
        with Path(file_path).open("r") as f:
            for line in f:
                if target_string in line:
                    return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_dino_finetuned_downloaded():
    # Load the original DINOv2 model with the correct architecture and parameters.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)  # Removed map_location
    # Load finetuned weights
    pretrained = torch.load(DINO_PATH_FINETUNED_DOWNLOADED, map_location=device)
    # Make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    # Change shape of pos_embed
    pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    model.pos_embed = pos_embed
    # Load state dict
    model.load_state_dict(new_state_dict, strict=True)
    # Move model to GPU
    model.to(device)
    return model

model = get_dino_finetuned_downloaded()

def preprocess_image(image):
    #Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    # image is a PIL Image
    return transform(image).unsqueeze(0)  # Add batch dimension



def preprocess_frame(frame):
    """Preprocess a single frame"""
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(frame)
    return transform(image)[:3]  # Ensure only RGB channels are considered

def video_to_embeddings(video_path, output_folder, done_file_path, batch_size=128):
    #"""Extract frames from a video and compute embeddings in batches"""
    try:
        vr = VideoReader(video_path, width=224, height=224)
    except:
        print(f'path doesnt exist: {video_path}')
        return
    total_frames = len(vr)
    all_embeddings = []

    for idx in range(0, total_frames, batch_size):
        batch_frames = vr.get_batch(range(idx, min(idx + batch_size, total_frames))).asnumpy()
        
        # Preprocess and stack frames to form a batch
        batch_tensors = torch.stack([preprocess_frame(frame) for frame in batch_frames]).cuda()

        with torch.no_grad():
            # Process the entire batch through the model
            batch_embeddings = model(batch_tensors.to('cuda')).cpu().numpy()
        
        all_embeddings.append(batch_embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)


    video_name = video_path.split('/')[-1].rsplit('.', 1)[0]
    np_path = f"{output_folder}/{video_name}.npy"
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    args = parser.parse_args()
    index = args.index

    """
    #H2S_HANDS 0..125 (parallalize to submit 256 jobs)
    #CHECK THE DINO MODEL !!!!!!!!!!!!!!!!!!!!!!!!
    fixed_list = load_file("/path/to/a/list/containing/all/the/hand/videos/hands.list")
    done_file_path = "/path/to/text/containing/the/completed/videos/doha_emb1.txt"
    problem_file_path = "/path/to/text/containing/the/problematic/videos/problem1.txt"
    output_folder = "/path/to/where/to/save/hand/embeddings/clips_emb_1m"
    batch_size_in = 650
    """
    
    """
    #H2S_FACES 0..125 (parallalize to submit 256 jobs)
    #CHECK THE DINO MODEL !!!!!!!!!!!!!!!!!!!!!!!!
    fixed_list = load_file("/path/to/a/list/containing/all/the/face/videos/faces.list")
    done_file_path = "/path/to/text/containing/the/completed/videos/dofa_emb1.txt"
    problem_file_path = "/path/to/text/containing/the/problematic/videos/problem1_face.txt"
    output_folder = "/path/to/where/to/save/face/embeddings/clips_emb_1m"
    batch_size_in = 325
    """


    """
    #YASL_HANDS : 0..125 (parallalize to submit 256 jobs)
    fixed_list = load_file("/path/to/a/list/containing/all/the/hand/videos/hands.list")
    done_file_path = "/path/to/text/containing/the/completed/videos/doha_emb1.txt"
    problem_file_path = "/path/to/text/containing/the/problematic/videos/problem1.txt"
    output_folder = "/path/to/where/to/save/hand/embeddings/clips_emb_1m"
    batch_size_in = 10_000
    """

    """
    #YASL_FACES: 0..125
    fixed_list = load_file("/path/to/a/list/containing/all/the/face/videos/faces.list")
    done_file_path = "/path/to/text/containing/the/completed/videos/dofa_emb1.txt"
    problem_file_path = "/path/to/text/containing/the/problematic/videos/problem1_face.txt"
    output_folder = "/path/to/where/to/save/face/embeddings/clips_emb_1m"
    batch_size_in = 5_000
    """    

    video_batches = [fixed_list[i:i + batch_size_in] for i in range(0, len(fixed_list), batch_size_in)]
    
    
    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for video_path in video_batches[index]:
        video_name = video_path.split('/')[-1].rsplit('.', 1)[0]
        np_path = f"{output_folder}/{video_name}.npy"
        
        if Path(np_path).exists():
            continue
        else:
            video_to_embeddings(video_path, output_folder, done_file_path, batch_size=512)


#Instructions: 
# 1. Edit the DINO path i.e. face or hands
# 2. Chose one setting. Yasl-hands , etc