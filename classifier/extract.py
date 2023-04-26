import torch
from torchvision import models

import os 
import glob
import argparse
from ResAttention import *

from train_evaluate import extract
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", help="number of predicted classes",
                    type=int, default="3")
parser.add_argument("--video_dir", help="video directory",
                    type=str, default="./videos/Train3/")
parser.add_argument("--save_dir", help="saving directory of extracted features",
                    type=str, default="./features/train3/")

args = parser.parse_args()

num_classes = args.num_classes
print(num_classes)
print(args.video_dir)

# model = ResidualAttentionModel_92(num_classes=args.num_classes, feat_ext=True)
#model = model.load_state_dict(torch.load(args.weight_dir))
# model.load_state_dict(torch.load(args.weight_dir))
model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
model = model.to(device)

model.fc = torch.nn.Identity(2048)


print("Extraction progress")
print("=" * 20)
set_list = os.listdir(args.video_dir)
for path in set_list:
    vids = []
    root = args.video_dir + path + '/*.mp4'
    for filename in glob.glob(root):
        vids.append(filename)
    save_dir = args.save_dir + path
    os.makedirs(save_dir, exist_ok=True)
    extract(model=model, dataloader=vids, save_dir=save_dir)
