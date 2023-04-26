# import torch
# from torchvision import datasets
# import torch.nn as nn
# import torch.optim as optim
import os 
import numpy as np
import argparse
from PIL import Image
# import cv2

import subprocess




parser = argparse.ArgumentParser()

parser.add_argument("--setType", help="typeToExtrac",
                    type=str, default="./images/Train")
args = parser.parse_args()




def extract(model, video_dir, save_dir):

    for emotion_class in os.listdir(video_dir):
        frame_path = f"{save_dir}/{emotion_class}"
        print('frame path: ' + frame_path)
        
        if os.path.exists(save_dir + '/' +  emotion_class):
            continue
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        
        for vid in os.listdir(video_dir + '/' +  emotion_class):
            # cap = cv2.VideoCapture(vid)
            videoPath = f"{video_dir}/{emotion_class}/{vid}"
            vname = vid.split('/')[-1] # Split filename from path (./hello/video.mp4 => video.mp4)
            vname = vname.split('.')[0] # Cutoff file extension (video.mp4 => video)
            frameCount = 0

            newFilename = f"{frame_path}/{vname}_%0d.jpg"
            # subprocess.run('ffmpeg')
            # print(['ffmpeg', '-i', videoPath, '-vf', "select=not(mod(n\,5))", '-vsync', 'vfr', '-q:v', 2 ,newFilename])
            subprocess.run(['ffmpeg', '-hide_banner',  '-loglevel', 'error' , '-i', videoPath, '-vf', "select=not(mod(n\,5))", '-vsync', 'vfr', '-q:v', '2' ,newFilename])
                # ret, frame = cap.read()
                # if not ret:
                #     break
                # if frameCount % 5 == 0:
                #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     im_pil = Image.fromarray(img)
                #     im_pil.save(newFilename)
                    
                # frameCount += 1
            # cap.release()

import glob
import cv2

def extractcv2(setType):

    for vid in glob.glob(f'./videos/{setType}/*/*.mp4'):
        cap = cv2.VideoCapture(vid)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        vname = vid.split('/')[-1] # Split filename from path (./hello/video.mp4 => video.mp4)
        className = vid.split('/')[-2]
        vname = vname.split('.')[0] # Cutoff file extension (video.mp4 => video)
        res = []

        frame_count = 0
        save_interval = 0.5
        save_dir = f'./images/cv2/Nine/{setType}/{className}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % (fps * save_interval) == 0:            
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)    
                im_pil.save(f'{save_dir}/{vname}_{frame_count}.jpg')

            frame_count += 1
        cap.release()

            

# extract(0, args.video_dir, args.save_dir)
extractcv2(args.setType)