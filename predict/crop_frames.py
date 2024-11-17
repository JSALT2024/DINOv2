import cv2
import numpy as np
import decord
import json


def crop_frame(image, bounding_box):
    x0, y0, x1, y1 = bounding_box
    cropped_frame = image[y0:y1, x0:x1]
    return cropped_frame


def resize_frame(frame, frame_size):
    if frame is not None and frame.size > 0:
        return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    else:
        return None


def get_local_crops(clip_path):
    json_path = f"{clip_path}.json"
    clip_path = f"{clip_path}.mp4"


    try:
        video = decord.VideoReader(clip_path)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error: {clip_path}")
        return

    with open(json_path, 'r') as rd:
        result_dict = json.load(rd)

    prev_face_frame = None
    prev_hand1_frame = None
    prev_hand2_frame = None

    out_face = []
    out_hand1 = []
    out_hand2 = []

    # check if video and keypoints have same number of frames
    frames = len(video)
    keypoints = len(result_dict["bbox_face"])
    if frames != keypoints:
        print("frames", frames)
        print("keypoints", keypoints)
        print(clip_path)
        print(json_path)

    for i in range(np.min([frames, keypoints])):
        frame = video[i].asnumpy()
        video.seek(0)

        # it contains a body pose that can be use as reference to get the face and hands
        if result_dict["bbox_face"][i]:
            face_bbox = result_dict["bbox_face"][i]
            face_frame = crop_frame(frame, face_bbox)
            face_frame = resize_frame(face_frame, (56, 56))
            out_face.append(face_frame)
            prev_face_frame = face_frame
        elif prev_face_frame is not None:
            out_face.append(prev_face_frame)
        else:
            face_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_face.append(face_frame)

        if result_dict["bbox_left_hand"][i]:
            hand1_bbox = result_dict["bbox_left_hand"][i]
            hand1_frame = crop_frame(frame, hand1_bbox)
            hand1_frame = resize_frame(hand1_frame, (56, 56))
            out_hand1.append(hand1_frame)
            prev_hand1_frame = hand1_frame
        elif prev_hand1_frame is not None:
            out_hand1.append(prev_hand1_frame)
        else:
            hand1_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand1.append(hand1_frame)

        if result_dict["bbox_right_hand"][i]:
            hand2_bbox = result_dict["bbox_right_hand"][i]
            hand2_frame = crop_frame(frame, hand2_bbox)
            hand2_frame = resize_frame(hand2_frame, (56, 56))
            out_hand2.append(hand2_frame)
            prev_hand2_frame = hand2_frame
        elif prev_hand2_frame is not None:
            out_hand2.append(prev_hand2_frame)
        else:
            hand2_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand2.append(hand2_frame)

    return out_face, out_hand1, out_hand2
