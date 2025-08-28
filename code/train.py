import os
import sys
import argparse
import json
from Evaluation import evaluation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import datasetLoader
from tqdm import tqdm

sys.path.append("../")

parser = argparse.ArgumentParser()

# Training Args
#   Constant, in our experiments:
parser.add_argument('-batchSize', type=int, default=20) # always 20
parser.add_argument('-nEpochs', type=int, default=50) # always 50
parser.add_argument('-nClasses', default= 2,type=int) # always 2 (bonafide, spoof)
parser.add_argument('-device', default= 'cuda',type=str) # always cuda

#   Variable
parser.add_argument('-alpha', required=False, default=0.5,type=float) # Determines balance between CYBORG and Cross Entropy losses. At 0.0, loss is entirely led by CYBORG/CAM alignment. At 1.0, loss is entirely led by cross entropy. In our paper, we try alphas {0.1, 0.3, 0.5, 0.7, 0.9} for all configurations.
parser.add_argument('-network', default= 'resnet',type=str) # CNN backbone, we try resnet, densenet, and inception for all configurations.

# Data Args
parser.add_argument('-csvPath', required=True, type=str) # with columns: split {"train", "test"}, truth {"Live", "Spoof"}, img_path, saliency_path
parser.add_argument('-outputPath', required=False, default = './model_output/',type=str)

args = parser.parse_args()

device = torch.device(args.device)

# CYBORG loss function hook
activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output
  return hook

# Definition of model architecture
if args.network == "resnet":
    im_size = 224
    map_size = 7
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)
    model.layer4[-1].conv3.register_forward_hook(getActivation('features'))
elif args.network == "inception":
    im_size = 299
    map_size = 8
    model = models.inception_v3(pretrained=True)
    model.aux_logits = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)
    model.Mixed_7c.register_forward_hook(getActivation('features'))
elif args.network == "densenet":
    im_size = 224
    map_size = 7
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)
else:
    print("Invalid selection...exiting.")
    sys.exit()

# Create destination folder
os.makedirs(args.outputPath, exist_ok=True)
log_path = os.path.join(args.outputPath, 'Logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)
result_path = os.path.join(args.outputPath , 'Results')
if not os.path.exists(result_path):
    os.mkdir(result_path)

class_to_id = {
    'Live': 0,
    'Spoof': 1
}

# Dataloader for train and test data
train_dataset = datasetLoader(
    split_file = args.csvPath,
    train_test = 'train',
    class_to_id = class_to_id,
    map_size = map_size,
    im_size = im_size
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)

test_dataset = datasetLoader(
    split_file = args.csvPath,
    train_test = 'test',
    class_to_id = class_to_id,
    map_size = map_size,
    im_size = im_size
)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)

dataloader = {
    'train': train_dataloader, 
    'test': test_dataloader
}

# Description of hyperparameters
lr = 0.005
solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)

criterion_xent = nn.CrossEntropyLoss()
criterion_hmap = nn.MSELoss()

# File for logging the training process
with open(os.path.join(log_path, 'params.json'), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}

train_loss=[]
test_loss=[]
bestAccuracy = 0
bestEpoch=0
alpha = args.alpha

print("Alpha value:",alpha)
if alpha == 1.0:
    print("Only using classification loss")
elif alpha == 0.0:
    print("Only using CYBORG loss")
else:
    print("Using classification loss and heatmap loss")

train_step = 0
val_step = 0
for epoch in range(args.nEpochs):
    for phase in ['train', 'test']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
        else:
            model.eval()

        tloss = 0.
        acc = 0.
        tot = 0
        c = 0
        testPredScore = []
        testTrueLabel = []
        imgNames=[]
        with torch.set_grad_enabled(train):
            for batch_idx, (data, cls, imageName, hmap) in enumerate(tqdm(dataloader[phase])):

                # Data and ground truth
                data = data.to(device)
                cls = cls.to(device)
                hmap = hmap.to(device)

                outputs = model(data)

                # Prediction of accuracy
                pred = torch.max(outputs,dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                class_loss = criterion_xent(outputs, cls)

                # Running model over data
                if phase == 'train' and alpha != 1.0:
                    if args.network == "densenet":
                        features = model.features(data)
                        params = list(model.classifier.parameters())[0]
                    elif args.network == "inception":
                        features = activation['features']
                        params = list(model.fc.parameters())[0]
                    elif args.network == "resnet":
                        features = activation['features']
                        params = list(model.fc.parameters())[0]
                    else:
                        print("INVALID ARCHITECTURE:",args.network)
                        sys.exit()

                    # features = activation['features']
                    bz, nc, h, w = features.shape

                    beforeDot =  features.reshape((bz, nc, h*w))
                    cams = []
                    for ids,bd in enumerate(beforeDot):
                        weight = params[cls[ids]]
                        # weight = params[pred[ids]]
                        cam = torch.matmul(weight, bd)
                        cam_img = cam.reshape(h, w)
                        cam_img = cam_img - torch.min(cam_img)
                        if torch.max(cam_img) != 0:
                            cam_img = cam_img / torch.max(cam_img)
                        cams.append(cam_img)
                        # if epoch == 0:
                        #     os.makedirs('/scratch365/aboyd3_new/CYBORG-Iris/cam_visualizations/model_training/',exist_ok=True)
                        #     cv2.imwrite('/scratch365/aboyd3_new/CYBORG-Iris/cam_visualizations/model_training/' + imageName[ids].replace(".jpg",".png"),cam_img.detach().cpu().numpy()*255)

                    cams = torch.stack(cams)
                    hmap_loss = (criterion_hmap(cams,hmap))
                else:
                    hmap_loss = 0

                # Optimization of weights for training data
                if phase == 'train':
                    if alpha != 1.0:
                        loss = (alpha)*(class_loss) + (1-alpha)*(hmap_loss)
                    else:
                        loss = class_loss
                    train_step += 1
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                elif phase == 'test':
                    loss = class_loss
                    val_step += 1
                    temp = outputs.detach().cpu().numpy()
                    scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                    testPredScore.extend(scores)
                    testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                    imgNames.extend(imageName)

                tloss += loss.item()
                c += 1

        # Logging of train and test results
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot)
            train_loss.append(tloss / c)

        elif phase == 'test':
            log['validation'].append(tloss / c)
            log['val_acc'].append(acc / tot)
            print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)
            lr_sched.step()
            test_loss.append(tloss / c)
            accuracy = acc / tot
            if (accuracy >= bestAccuracy):
                bestAccuracy =accuracy
                testTrueLabels = testTrueLabel
                testPredScores = testPredScore
                bestEpoch = epoch
                save_best_model = os.path.join(log_path,'final_model.pth')
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': solver.state_dict(),
                }
                torch.save(states, save_best_model)
                testImgNames= imgNames

    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': solver.state_dict(),
    }
    with open(os.path.join(log_path,'model_log.json'), 'w') as out:
        json.dump(log, out)
    torch.save(states, os.path.join(log_path,'current_model.pth'))


# Plotting of train and test loss
plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
plt.plot(np.arange(0, args.nEpochs), test_loss[:], 'b')
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(os.path.join(result_path,'model_Loss.jpg'))


# Evaluation of test set utilizing the trained model
obvResult = evaluation()
errorIndex, predictScore, threshold = obvResult.get_result(args.network, testImgNames, testTrueLabels, testPredScores, result_path)