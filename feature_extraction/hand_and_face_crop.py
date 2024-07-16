import cv2
import numpy as np
import os
import pickle
import gzip
from datetime import datetime
from pathlib import Path
import decord
import argparse
import json

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

def resize_frame(frame, frame_size):
    if frame is not None and frame.size > 0:
        return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    else:
        return None

def crop_frame(image, bounding_box):
    x, y, w, h = bounding_box
    cropped_frame = image[y:y + h, x:x + w]
    return cropped_frame

def get_bounding_box(landmarks, image_shape, scale_factor=1.2):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    x, y, w, h = cv2.boundingRect(landmarks_px)
    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding
    return x, y, w, h

def adjust_bounding_box(bounding_box, image_shape):
    x, y, w, h = bounding_box
    ih, iw, _ = image_shape
    """
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > iw:
        w = iw - x
    if y + h > ih:
        h = ih - y
    """
    # Adjust x-coordinate if the bounding box extends beyond the image's right edge
    if x + w > iw:
        x = iw - w
    
    # Adjust y-coordinate if the bounding box extends beyond the image's bottom edge
    if y + h > ih:
        y = ih - h
    
    # Ensure bounding box's x and y coordinates are not negative
    x = max(x, 0)
    y = max(y, 0)    

    return x, y, w, h

def get_centered_box(landmarks, image_shape, box_size, scale_factor=1.5):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size

    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding

    return x, y, w, h


def is_center_inside_frame(landmarks, image_shape):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    return 0 <= center_x <= iw and 0 <= center_y <= ih


def isl_wrist_below_elbow(pose_landmarks):
    left_elbow = pose_landmarks[13]
    left_wrist = pose_landmarks[15]
    return left_wrist[1] > left_elbow[1]

def isr_wrist_below_elbow(pose_landmarks):
    right_elbow = pose_landmarks[14]
    right_wrist = pose_landmarks[16]
    return right_wrist[1] > right_elbow[1]    

def get_hand_direction(elbow_landmark, wrist_landmark):
    direction = np.array([wrist_landmark[0], wrist_landmark[1]]) - np.array([elbow_landmark[0], elbow_landmark[1]])
    direction = direction / np.linalg.norm(direction)
    return direction

def shift_bounding_box(bounding_box, direction, magnitude):
    x, y, w, h = bounding_box
    shift_x = int(direction[0] * magnitude)
    shift_y = int(direction[1] * magnitude)
    shifted_box = (x + shift_x, y + shift_y, w, h)
    return shifted_box

def select_face(pose_landmarks, face_landmarks):
    # select the nose landmark from the pose landmark.
    nose_landmark_from_pose = pose_landmarks[0]
    # list of nose landamrk(s) obtained from the face landmarks
    nose_landmarks_from_face = []
    for i in range(0, len(face_landmarks)):
        nose_landmarks_from_face.append(face_landmarks[i][0])
    # return the indices of the closest nose from nose_landmarks_from_face to nose_landmark_from_pose
    closest_nose_index = np.argmin([np.linalg.norm(np.array(nose_landmark_from_pose) - np.array(nose_landmark)) for nose_landmark in nose_landmarks_from_face])
    return face_landmarks[closest_nose_index]

def select_hands(pose_landmarks, hand_landmarks, image_shape):
    
    if hand_landmarks is None:
        return None, None
    # select the wrist landmarks from the pose landmark.
    left_wrist_from_pose = pose_landmarks[15]
    #print(f'left wrist from pose: {left_wrist_from_pose}')
    right_wrist_from_pose = pose_landmarks[16]
    #print(f'right wrist from pose: {right_wrist_from_pose}')
    # check if these coordinates are inside the frame
    ih, iw, _ = image_shape        
        
    # if (0 <= right_wrist_from_pose[0] <= iw and 0 <= right_wrist_from_pose[1] <= ih) and (0 <= pose_landmarks[18][0] <= iw and 0 <= pose_landmarks[18][1] <= ih) and (0 <= pose_landmarks[20][0] <= iw and 0 <= pose_landmarks[20][1] <= ih) and (0 <= pose_landmarks[22][0] <= iw and 0 <= pose_landmarks[22][1] <= ih):         
    #     right_wrist_from_pose = pose_landmarks[16]
    # else:
    #     right_wrist_from_pose = None
        
    # if (0 <= left_wrist_from_pose[0] <= iw and 0 <= left_wrist_from_pose[1] <= ih) and (0 <= pose_landmarks[17][0] <= iw and 0 <= pose_landmarks[17][1] <= ih) and (0 <= pose_landmarks[19][0] <= iw and 0 <= pose_landmarks[19][1] <= ih) and (0 <= pose_landmarks[21][0] <= iw and 0 <= pose_landmarks[21][1] <= ih):
    #     left_wrist_from_pose = pose_landmarks[15]
    # else:
    #     left_wrist_from_pose = None
    
    wrist_from_hand = []    
    for i in range(0, len(hand_landmarks)):
        # array of wrist landmarks from the hand landmarks
        wrist_from_hand.append(hand_landmarks[i][0])
    
    # the euclidean distance between the two points using only the first 2 coordinates.
    if right_wrist_from_pose is not None:
        right_hand_landmarks = hand_landmarks[0]
        minimum_distance = 100
        for i in range(0, len(hand_landmarks)):
            distance = np.linalg.norm(np.array(right_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
            if distance < minimum_distance:
                minimum_distance = distance
                right_hand_landmarks = hand_landmarks[i]
        #print(f'right wrist distance: {minimum_distance}')
        if minimum_distance >= 0.1:
            right_hand_landmarks = None
            
    else:
        #print("right wrist is distance: 0")
        right_hand_landmarks = None
    
    if left_wrist_from_pose is not None:
        left_hand_landmarks = hand_landmarks[0]
        minimum_distance = 100
        for i in range(0, len(hand_landmarks)):
            distance = np.linalg.norm(np.array(left_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
            if distance < minimum_distance:
                minimum_distance = distance
                left_hand_landmarks = hand_landmarks[i]
        #print(f'left wrist distance: {minimum_distance}')
        if minimum_distance >= 0.1:
            left_hand_landmarks = None

    else:
        #print("left wrist is distance: 0")
        left_hand_landmarks = None
        
    return left_hand_landmarks, right_hand_landmarks
    
   
        
    


def video_holistic(video_file, face_path, hand_path, done_file_path, problem_file_path, pose_path, stats_path):
    try:
        video = decord.VideoReader(video_file)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error: {video_file}")
        with (Path(problem_file_path)).open("a") as p:
            p.write(video_file + "\n")
        return

    fps = video.get_avg_fps()

    clip_face_path = f"{face_path}{video_file.split('/')[-1].rsplit('.', 1)[0]}_face.mp4"
    clip_hand1_path = f"{hand_path}{video_file.split('/')[-1].rsplit('.', 1)[0]}_hand1.mp4"
    clip_hand2_path = f"{hand_path}{video_file.split('/')[-1].rsplit('.', 1)[0]}_hand2.mp4"
    landmark_json_path = Path(f"{pose_path}{video_file.split('/')[-1].rsplit('.', 1)[0]}_pose.json")

    fourcc_face = cv2.VideoWriter_fourcc(*'mp4v')
    out_face = cv2.VideoWriter(clip_face_path, fourcc_face, fps, (56, 56))

    fourcc_hand1 = cv2.VideoWriter_fourcc(*'mp4v')
    out_hand1 = cv2.VideoWriter(clip_hand1_path, fourcc_hand1, fps, (56, 56))

    fourcc_hand2 = cv2.VideoWriter_fourcc(*'mp4v')
    out_hand2 = cv2.VideoWriter(clip_hand2_path, fourcc_hand2, fps, (56, 56))

    with open(landmark_json_path, 'r') as rd:
        result_dict = json.load(rd)

    prev_face_frame = None
    prev_hand1_frame = None
    prev_hand2_frame = None
    prev_result_dict = None

    face_box_size = None
    hand1_box_size = None
    hand2_box_size = None
    
    #print result_dict
    #print(f"Result_dict: {result_dict}")

    for i in range(len(video)):
        frame = video[i].asnumpy()
        #print(f"Frame {i} shape: {frame.shape}")
        video.seek(0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # length of result_dict
        #print(f"Length of result_dict: {len(result_dict)}")
        if result_dict[str(i)] is None: # no pose_landmarks detected
            if prev_result_dict is not None: # use the previous pose_landmarks
                result_dict[str(i)] = prev_result_dict
            else: # use a blank frame if no pose_landmarks was ever detected for this clip
                continue
        else: # store pose_landmarks detected as last known pose_landmarks
            prev_result_dict = result_dict[str(i)]

        if result_dict[str(i)]['pose_landmarks'] is None: # some pose_landmarks were detected
            if prev_face_frame is not None:
                out_face.write(prev_face_frame)
            else:
                face_frame = np.zeros((56, 56, 3), dtype=np.uint8)
                out_face.write(face_frame)
            if prev_hand1_frame is not None:
                out_hand1.write(prev_hand1_frame)
            else:
                hand1_frame = np.zeros((56, 56, 3), dtype=np.uint8)
                out_hand1.write(hand1_frame)
            if prev_hand2_frame is not None:
                out_hand2.write(prev_hand2_frame)   
            else:
                hand2_frame = np.zeros((56, 56, 3), dtype=np.uint8)
                out_hand2.write(hand2_frame)
                
            continue
        
                       
        # it contains a body pose that can be use as reference to get the face and hands   
        if result_dict[str(i)]['face_landmarks'] is not None: # some face_landmarks were detected
            face_lmks = select_face(result_dict[str(i)]['pose_landmarks'][0], result_dict[str(i)]['face_landmarks']) # from all the faces, select the one that is closer to the face of the pose landmark
            face_box = get_bounding_box(face_lmks, frame_rgb.shape)
            face_box = adjust_bounding_box(face_box, frame_rgb.shape) # to make sure it is within the frame
            face_frame = crop_frame(frame_rgb, face_box)
            face_frame = resize_frame(face_frame, (56, 56))            
            out_face.write(face_frame)
            prev_face_frame = face_frame

        elif prev_face_frame is not None:
            out_face.write(prev_face_frame) 
        else:
            # use a blank frame if no face_landmarks or body_landmarks were detected
            face_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_face.write(face_frame)
            #continue 
        
        
        
        # select the left and right hand from the hand_landmarks
        result_dict[str(i)]['left_hand_landmarks'], result_dict[str(i)]['right_hand_landmarks'] = select_hands(result_dict[str(i)]['pose_landmarks'][0], result_dict[str(i)]['hand_landmarks'], frame_rgb.shape)
        #print(len(result_dict[str(i)]['hand_landmarks']))
        #result_dict[str(i)]['left_hand_landmarks'], result_dict[str(i)]['right_hand_landmarks'] = result_dict[str(i)]['hand_landmarks'][0], result_dict[str(i)]['hand_landmarks'][1]
        

        if result_dict[str(i)]['left_hand_landmarks'] is not None:
            hand1_box = get_bounding_box(result_dict[str(i)]['left_hand_landmarks'], frame_rgb.shape, scale_factor=1.2)
            hand1_box = adjust_bounding_box(hand1_box, frame_rgb.shape)
            hand1_frame = crop_frame(frame_rgb, hand1_box)
            hand1_frame = resize_frame(hand1_frame, (56, 56))
            out_hand1.write(hand1_frame)
            prev_hand1_frame = hand1_frame                       
        elif prev_hand1_frame is not None:
            out_hand1.write(prev_hand1_frame)
        else:
            hand1_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand1.write(hand1_frame)

        if result_dict[str(i)]['right_hand_landmarks'] is not None:
            hand2_box = get_bounding_box(result_dict[str(i)]['right_hand_landmarks'], frame_rgb.shape, scale_factor=1.2)
            hand2_box = adjust_bounding_box(hand2_box, frame_rgb.shape)
            hand2_frame = crop_frame(frame_rgb, hand2_box)
            hand2_frame = resize_frame(hand2_frame, (56, 56))
            out_hand2.write(hand2_frame)
            prev_hand2_frame = hand2_frame                       
        elif prev_hand2_frame is not None:
            out_hand2.write(prev_hand2_frame)
        else:
            hand2_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand2.write(hand2_frame)

    with (Path(done_file_path)).open("a") as f:
        f.write(video_file + "\n")

    out_face.release()
    out_hand1.release()
    out_hand2.release()
    del out_face
    del out_hand1
    del out_hand2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    args = parser.parse_args()
    index = args.index

    # h2s
    fixed_list = load_file("/shester/crops2/face/h2s/000/train_dev_test.list")
    done_file_path = "/shester/crops2/face/h2s/000/done_crop.txt"
    problem_file_path = "/shester/crops2/face/h2s/000/problem_crop.txt"
    pose_path = "/shester/crops2/face/h2s/pose2/"
    stats_path = "/shester/crops2/face/h2s/stats2/"
    face_path = "/shester/crops2/face/h2s/clips64/"
    hand_path = "/shester/crops2/hand/h2s/clips64/"
    batch_size = 150 

    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    for video_file in video_batches[index]:
        
        clip_hand2_path = f"{hand_path}{video_file.split('/')[-1].rsplit('.', 1)[0]}_hand2.mp4"
        
        if os.path.exists(clip_hand2_path):
            continue
        elif is_string_in_file(problem_file_path, video_file):
            continue
        else:
            video_holistic(video_file, face_path, hand_path, done_file_path, problem_file_path, pose_path, stats_path)