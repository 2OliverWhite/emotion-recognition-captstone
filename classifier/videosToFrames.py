# import torch
# from torchvision import datasets
# import torch.nn as nn
# import torch.optim as optim
import os 
import numpy as np
import argparse
from PIL import Image
# import cv2




parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", help="number of predicted classes",
                    type=int, default="3")
parser.add_argument("--batch_size", help="batch size of cnn",
                    type=int, default="16")
parser.add_argument("--num_epochs", help="training epochs",
                    type=int, default="100")
parser.add_argument("--train_root", help="training data directory",
                    type=str, default="../images/train_three/")
parser.add_argument("--valid_root", help="validation data directory",
                    type=str, default="../images/train_three/")
parser.add_argument("--video_dir", help="src of videos to extract frames from",
                    type=str, default="./data/videos")
parser.add_argument("--save_dir", help="place to put extracted frames",
                    type=str, default="./data/images")
args = parser.parse_args()


print(os.listdir(args.video_dir))
def extract(model, video_dir, save_dir):

    for emotion_class in os.listdir(video_dir):
        frame_path = f"{save_dir}/{emotion_class}"
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        for vid in os.listdir(video_dir + '/' +  emotion_class):
            print(f"Vid is {vid}")
            # cap = cv2.VideoCapture(vid)
            vname = vid.split('/')[-1] # Split filename from path (./hello/video.mp4 => video.mp4)
            vname = vname.split('.')[0] # Cutoff file extension (video.mp4 => video)
            frameCount = 0
            while True:
                print(vname)
                break
                # ret, frame = cap.read()
                # if not ret:
                #     break
                #if frameCount % 5 == 0:
                    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # im_pil = Image.fromarray(img)
                    
                #frameCount += 1
                # cap.release()
            
extract(0, args.video_dir, args.save_dir)