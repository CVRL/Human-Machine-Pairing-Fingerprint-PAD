import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import csv
import os
import csv
import argparse

sys.path.append("../")

if __name__ == '__main__':
    print("\nTESTING\n")

    parser = argparse.ArgumentParser()
    device = torch.device('cuda')
    
    parser.add_argument('-modelPath',type=str)
    parser.add_argument('-network',  default="resnet",type=str)
    parser.add_argument('-csv', type=str)
    parser.add_argument('-outfile', default="all.csv", type=str)
    args = parser.parse_args()

    if args.network == "resnet":
        im_size = 224
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.network == "inception":
        im_size = 299
        model = models.inception_v3(pretrained=True)
        model.aux_logits=False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.network == "densenet":
        im_size = 224
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    else:
        print("Invalid selection...exiting.")
        sys.exit()

    weights = torch.load(args.modelPath, map_location = device)
    model.load_state_dict(weights['state_dict'])

    model = model.to(device)
    model.eval()

    print("Model loaded")

    transform = transforms.Compose([
        transforms.Resize([im_size, im_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sigmoid = nn.Sigmoid()

    print("Processing csv")

    image_scores = []
    with open(args.csv, 'r') as f:
        reader = csv.reader(f)
        for split, truth, img_path, _ in reader:

            if split != 'test':
                continue
            
            if truth == "Live":
                truth = 0
            elif truth == "Spoof":
                truth = 1
            else:
                print(f"Error: unknown truth class value: {truth} for {img_path}")
                continue
            
            if not os.path.exists(img_path):
                print(f"Error: testing image not found: {img_path}")
                continue
            
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            img = img[0:3,:,:].unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                output = model(img)

            prediction = sigmoid(output).detach().cpu().numpy()[:, 1][0]
            image_scores.append([img_path, prediction, truth])

    print("Testing complete!")

    # Writing the scores in the csv output file
    with open(args.outfile, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(image_scores)

    print("Saved to: ", args.outfile)