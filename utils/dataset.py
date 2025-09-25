from moviepy.editor import VideoFileClip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import whisper
import tempfile
import os 
import numpy as np
import cv2


class PreExtractedFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(feature_dir)))}
        
        for cls in sorted(os.listdir(feature_dir)):
            cls_path = os.path.join(feature_dir, cls)
            for file in os.listdir(cls_path):
                # # if file.endswith('.pt'):
                # print(os.path.join(cls_path, file))
                # raise
                self.feature_paths.append(os.path.join(cls_path, file))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.feature_paths)

    def extract_frames(self, video_path, num_frames=30, resize=(224, 224)):
        """
        Trích xuất các khung hình từ video.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return np.zeros((num_frames, resize[0], resize[1], 3))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // num_frames, 1)

        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, resize)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if len(frames) == num_frames:
                break
        cap.release()

        while len(frames) < num_frames:
            frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2)) / 255.0
        frames = torch.tensor(frames, dtype=torch.float)

        return frames 

    def __getitem__(self, idx):
        frame_path = self.feature_paths[idx]
        # print(frame_path)
        frames = self.extract_frames(frame_path)
        label = self.labels[idx]

        return {
            'frames': frames,
            'label': label
        }

import logging
import os 
def logg(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )