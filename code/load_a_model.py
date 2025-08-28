import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms

def get_model(network):
    if network == "resnet":
        img_size = 224
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model, img_size
    elif network == "inception":
        img_size = 299
        model = models.inception_v3(pretrained=True)
        model.aux_logits=False
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model, img_size
    elif network == "densenet":
        img_size = 224
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model, img_size
    else:
        print("Invalid selection...exiting.")
        sys.exit()

# Simple example of loading a trained model and preparing it for inference
# See test.py for a more complete example

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

path_to_trained_model_weights = "path/to/your/model_weights.pth"
architecture_of_trained_model = "resnet" # or "densenet", or "inception"

model, img_size = get_model(architecture_of_trained_model)
    
weights = torch.load(path_to_trained_model_weights, map_location = device)
model.load_state_dict(weights['state_dict'])
model = model.to(device)
model.eval()

img_transform = transforms.Compose([
    transforms.Resize([img_size, img_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])