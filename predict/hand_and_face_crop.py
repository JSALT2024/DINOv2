import cv2
import numpy as np
import os
import decord
import json


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
    landmarks_px = np.round(np.array(landmarks)[:, :2]).astype(int)
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


def video_holistic(clip_name, clip_folder):
    clip_path = os.path.join(clip_folder, f"{clip_name}.mp4")
    json_path = os.path.join(clip_folder, f"{clip_name}.json")
    try:
        video = decord.VideoReader(clip_path)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error: {clip_path}")
        return

    with open(json_path, 'r') as rd:
        result_dict = json.load(rd)["joints"]

    prev_face_frame = None
    prev_hand1_frame = None
    prev_hand2_frame = None
    prev_result_dict = None

    out_face = []
    out_hand1 = []
    out_hand2 = []

    for i in range(len(video)):
        frame = video[i].asnumpy()
        video.seek(0)
        if result_dict[str(i)] is None:  # no pose_landmarks detected
            if prev_result_dict is not None:  # use the previous pose_landmarks
                result_dict[str(i)] = prev_result_dict
            else:  # use a blank frame if no pose_landmarks was ever detected for this clip
                continue
        else:  # store pose_landmarks detected as last known pose_landmarks
            prev_result_dict = result_dict[str(i)]

        if result_dict[str(i)]['pose_landmarks'] is None:  # some pose_landmarks were detected
            if prev_face_frame is not None:
                out_face.append(prev_face_frame)
            else:
                face_frame = np.zeros((56, 56, 3), dtype=np.uint8)
                out_face.append(face_frame)
            if prev_hand1_frame is not None:
                out_hand1.append(prev_hand1_frame)
            else:
                hand1_frame = np.zeros((56, 56, 3), dtype=np.uint8)
                out_hand1.append(hand1_frame)
            if prev_hand2_frame is not None:
                out_hand2.append(prev_hand2_frame)
            else:
                hand2_frame = np.zeros((56, 56, 3), dtype=np.uint8)
                out_hand2.append(hand2_frame)

            continue

        # it contains a body pose that can be use as reference to get the face and hands
        if result_dict[str(i)]['face_landmarks']:  # some face_landmarks were detected
            face_lmks = result_dict[str(i)]['face_landmarks']
            face_box = get_bounding_box(face_lmks, frame.shape)
            face_box = adjust_bounding_box(face_box, frame.shape)  # to make sure it is within the frame
            face_frame = crop_frame(frame, face_box)
            face_frame = resize_frame(face_frame, (56, 56))
            out_face.append(face_frame)
            prev_face_frame = face_frame
        elif prev_face_frame is not None:
            out_face.append(prev_face_frame)
        else:
            # use a blank frame if no face_landmarks or body_landmarks were detected
            face_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_face.append(face_frame)

        if result_dict[str(i)]['left_hand_landmarks']:
            hand1_box = get_bounding_box(result_dict[str(i)]['left_hand_landmarks'], frame.shape, scale_factor=1.2)
            hand1_box = adjust_bounding_box(hand1_box, frame.shape)
            hand1_frame = crop_frame(frame, hand1_box)
            hand1_frame = resize_frame(hand1_frame, (56, 56))
            out_hand1.append(hand1_frame)
            prev_hand1_frame = hand1_frame
        elif prev_hand1_frame is not None:
            out_hand1.append(prev_hand1_frame)
        else:
            hand1_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand1.append(hand1_frame)

        if result_dict[str(i)]['right_hand_landmarks']:
            hand2_box = get_bounding_box(result_dict[str(i)]['right_hand_landmarks'], frame.shape, scale_factor=1.2)
            hand2_box = adjust_bounding_box(hand2_box, frame.shape)
            hand2_frame = crop_frame(frame, hand2_box)
            hand2_frame = resize_frame(hand2_frame, (56, 56))
            out_hand2.append(hand2_frame)
            prev_hand2_frame = hand2_frame
        elif prev_hand2_frame is not None:
            out_hand2.append(prev_hand2_frame)
        else:
            hand2_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand2.append(hand2_frame)

    return out_face, out_hand1, out_hand2
