import os
import torch
import torch.utils.data as data_utl
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
import csv

class datasetLoader(data_utl.Dataset):
    def __init__(self, split_file, train_test, class_to_id, map_size=7, im_size=224):
        self.split_file = split_file
        self.train_test = train_test
        self.class_to_id = class_to_id
        self.map_size = map_size
        self.image_size = im_size

        self.data = []

        self.transform = transforms.Compose([
            transforms.Resize([self.image_size, self.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.saliency_transform = transforms.Compose([
            transforms.Resize([self.map_size, self.map_size]),
            transforms.ToTensor(),
        ])

        # Reading data from CSV file
        print(f"In data_loader_better.py, reading split: {train_test}")
        with open(split_file, 'r') as f:
            reader = csv.reader(f)
            for split, truth, img_path, saliency_path in reader:

                # If it isn't in the split we're reading (train != test, test != train)
                if split != train_test:
                    continue

                # Make sure image/saliency pair exists and is valid
                if truth not in self.class_to_id:
                    print(f"Unknown truth class encountered: {truth} for {img_path}")
                    return
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    return
                if not os.path.exists(saliency_path):
                    print(f"Saliency not found: {saliency_path}")
                    return

                # Load and transform image
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)

                # Load and transform saliency
                if split == "train":
                    saliency = Image.open(saliency_path).convert('L')
                    saliency = self.saliency_transform(saliency)
                    saliency = saliency.type(torch.float)
                    saliency = torch.squeeze(saliency)
                    saliency = saliency - torch.min(saliency)
                    if torch.max(saliency) != 0:
                        saliency = saliency / torch.max(saliency)
                else:
                    saliency = 0 # Just have no saliency for the testing samples. Can't return NoneType, but this won't be read anyways

                self.data.append([img_path, self.class_to_id[truth], img[0:3,:,:], saliency])

    def __getitem__(self, index):
        img_path, truth, img, saliency = self.data[index]
        return img, truth, img_path, saliency

    def __len__(self):
        return len(self.data)
