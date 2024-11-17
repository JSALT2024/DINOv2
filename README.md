
## Create dino feature h5 dataset
```shell
python predict/create_dino_features.py \
  --input_folder data/cropped_clips
  --output_folder data/features
  --dataset_name h2s \
  --split_name train \
  --annotation_file data\how2sign_realigned_train.csv   # only if the name is in wrong format
```


## Predict (embedding)
```Python
import sys
import cv2
import torch
sys.path.append('predict')
import predict_dino

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

face_checkpoint = ""
hand_checkpoint = ""
face_image_path = ""
left_image_path = ""
right_image_path = ""

face_model = predict_dino.create_dino_model(face_checkpoint)
hand_model = predict_dino.create_dino_model(hand_checkpoint)
face_model.to(device)
hand_model.to(device)

face_image = cv2.imread(face_image_path)
left_hand_image = cv2.imread(left_image_path)
right_hand_image = cv2.imread(right_image_path)

face_features = predict_dino.dino_predict(face_image, face_model, predict_dino.transform_dino, device)
left_features = predict_dino.dino_predict(left_hand_image, hand_model, predict_dino.transform_dino, device)
right_features = predict_dino.dino_predict(right_hand_image, hand_model, predict_dino.transform_dino, device)
features = np.concatenate([face_features, left_features, right_features], 1)
```