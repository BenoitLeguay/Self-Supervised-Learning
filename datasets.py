import torch
import os
import glob
from PIL import Image
import random
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
MEAN_NORM = [0.485, 0.456, 0.406]
STD_NORM = [0.229, 0.224, 0.225]


class CatsDogsRotateDatasets(torch.utils.data.Dataset):
    def __init__(self, sub_percent=None):
        self.dir = os.path.join(os.environ["CATS_DOGS"])
        self.cat_dir = os.path.join(self.dir, "Cat")
        self.dog_dir = os.path.join(self.dir, "Dog")
        self.images = glob.glob(os.path.join(self.cat_dir, "*")) + glob.glob(os.path.join(self.dog_dir, "*"))
        self.general_transform = transforms.Compose([
                            transforms.Resize([224, 224]),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN_NORM, STD_NORM)
        ])
        if sub_percent:
            self.images = self.images_sub_percent(sub_percent)

    def images_sub_percent(self, sub_percent):
        random.shuffle(self.images)
        n_elements = len(self)
        trunc_low, trunc_high = int(n_elements*sub_percent[0]), int(n_elements*sub_percent[1])
        return self.images[trunc_low: trunc_high]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.general_transform(image)
        label = 0
        if .5 > random.uniform(0, 1):
            image = transforms.functional.rotate(image, 90)
        else:
            image = transforms.functional.rotate(image, 180)
            label = 1

        return image, label


class CatsDogsDatasets(torch.utils.data.Dataset):
    def __init__(self, sub_percent=None):
        self.dir = os.path.join(os.environ["CATS_DOGS"])
        self.cat_dir = os.path.join(self.dir, "Cat")
        self.dog_dir = os.path.join(self.dir, "Dog")
        self.images_cats = glob.glob(os.path.join(self.cat_dir, "*"))
        self.images_dogs = glob.glob(os.path.join(self.dog_dir, "*"))
        self.images = self.images_cats + self.images_dogs
        self.general_transform = transforms.Compose([
                            transforms.Resize([224, 224]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if sub_percent:
            self.images = self.images_sub_percent(sub_percent)

    def images_sub_percent(self, sub_percent):
        random.shuffle(self.images)
        n_elements = len(self)
        trunc_low, trunc_high = int(n_elements*sub_percent[0]), int(n_elements*sub_percent[1])
        return self.images[trunc_low: trunc_high]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.general_transform(image)
        label = 0
        if self.images[index] in self.images_dogs:
            label = 1

        return image, label


def show_tensor_images(images_tensor, labels_tensor, label_names, n_images=10):

    images_tensor, labels_tensor = images_tensor[:n_images], labels_tensor[:n_images]

    images_tensor_unnormalized = images_tensor.new(*images_tensor.size())
    images_tensor_unnormalized[:, 0, :, :] = images_tensor[:, 0, :, :] * STD_NORM[0] + MEAN_NORM[0]
    images_tensor_unnormalized[:, 1, :, :] = images_tensor[:, 1, :, :] * STD_NORM[1] + MEAN_NORM[1]
    images_tensor_unnormalized[:, 2, :, :] = images_tensor[:, 2, :, :] * STD_NORM[2] + MEAN_NORM[2]

    _, axis = plt.subplots(1, n_images, figsize=(20, 20))
    for i, (img, label) in enumerate(zip(images_tensor_unnormalized, labels_tensor)):
        axis[i].imshow(img.permute(1, 2, 0))
        axis[i].xaxis.set_visible(False)
        axis[i].yaxis.set_visible(False)

        if label == 0:
            axis[i].set_title(label_names[0])
        else:
            axis[i].set_title(label_names[1])

