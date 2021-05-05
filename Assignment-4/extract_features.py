import os
import numpy as np
import scipy.io as sio
import glob
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from torchvision import models
import torch
# SPECIFY PATH TO THE DATASET
path_to_dataset = '../tiny-UCF101/'


def main():

    feature = []
    label = []
    categories = sorted(os.listdir(path_to_dataset))

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # FILL IN TO LOAD THE ResNet50 MODEL
    model = models.resnet50(pretrained=True)
    extractor = torch.nn.Sequential(*list(model.children())[: -1])
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    extractor = extractor.to(device)
    extractor.eval()
    for i, c in enumerate(categories):
        path_to_images = sorted(
            glob.glob(os.path.join(path_to_dataset, c) + '/*.jpg'))
        for p in path_to_images:
            # FILL IN TO LOAD IMAGE, PREPROCESS, EXTRACT FEATURES.
            # OUTPUT VARIABLE F EXPECTED TO BE THE FEATURE OF THE IMAGE OF DIMENSION (2048,)
            img = Image.open(p)
            img = transform_test(img)
            x = torch.unsqueeze(img, dim=(0))
            x = x.to(device)
            F = extractor(x).cpu()
            F = F.detach().numpy()
            feature.append(F[0])
            label.append(categories.index(c))
    sio.savemat('ucf101dataset.mat', mdict={
                'feature': feature, 'label': label})


if __name__ == "__main__":
    main()
