import os
import glob
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

random_state = 42


# class PH2(torch.utils.data.Dataset):
#     def __init__(self, train, transform, data_path='./PH2_Dataset_images'):
#         'Initialization'
#         self.transform = transform
#         data_path = os.path.join(data_path, 'train' if train else 'test')
#         image_classes = [os.path.split(d)[1] for d in glob.glob(
#             data_path + '/*') if os.path.isdir(d)]
#         image_classes.sort()
#         self.name_to_label = {c: id for id, c in enumerate(image_classes)}
#         self.image_paths = glob.glob(data_path + '/*/*.jpg')

#     def __len__(self):
#         'Returns the total number of samples'
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         'Generates one sample of data'
#         image_path = self.image_paths[idx]

#         image = Image.open(image_path)
#         c = os.path.split(os.path.split(image_path)[0])[1]
#         y = self.name_to_label[c]
#         X = self.transform(image)
#         return X, y

class PH2(torch.utils.data.Dataset):
    def __init__(self, transform, image_paths, label_paths):
        'Initialization'
        self.transform = transform
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y


def load_and_transform_dataset(train_size, test_size, batch_size, data_path='/content/drive/MyDrive/datasets/hotdog_nothotdog'):

    min_scale = 256
    max_scale = 480
    
    target_size = (150, 150)
    # target_size = (image_resize, image_resize)

    normalize = transforms.Normalize(
        mean=[0.5244, 0.4443, 0.3621],
        std=[0.2679, 0.2620, 0.2733],
    )

    transform_list = []

    # if rand_resize:
    #     transform_list.append(transforms.Resize(
    #         random.randint(min_scale, max_scale)))

    # if rand_crop:
    #     transform_list.append(transforms.RandomCrop(target_size))

    # if target_resize:
    #     transform_list.append(transforms.Resize(target_size))

    # if flip:
    #     transform_list.append(transforms.RandomHorizontalFlip())

    # if rotate:
    #     transform_list.append(transforms.RandomRotation(10))

    # transform_list.append(transforms.ToTensor())
    # transform_list.append(normalize)

    # train_transform = transforms.Compose(transform_list)

    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        # normalize,
    ])
    

    image_paths = glob.glob(data_path + '/images/*')
    label_paths = glob.glob(data_path + '/masks/*')
    

    train_img_paths, test_img_paths, train_label_paths, test_label_paths = train_test_split(
        image_paths, label_paths, test_size=test_size, random_state=random_state)

    train_img_paths, val_img_paths, train_label_paths, val_label_paths = train_test_split(
        train_img_paths, train_label_paths, test_size=test_size, random_state=random_state)
    
    

    trainset = PH2(transform=test_transform,
                   image_paths=train_img_paths, label_paths=train_label_paths)
    valset = PH2(transform=test_transform,
                  image_paths=val_img_paths, label_paths=val_label_paths)
    testset = PH2(transform=test_transform,
                  image_paths=test_img_paths, label_paths=test_label_paths)
    
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=3)

    return trainset, valset, testset, train_loader, val_loader, test_loader


#  trainset, testset, val_loader, val_loader, train_loader, test_loader = load_and_transform_dataset(0.8, 0.1, 16, './DRIVE/training/')
# trainset, valset, testset, train_loader, val_loader, test_loader = load_and_transform_dataset(0.8, 0.1, 16, './PH2_Dataset_images/')

# print(len(trainset))
# print(len(valset))
# print(len(testset))
# # print(len(trainset))


# img, label = trainset[0]

# print(img.shape)
# print(label.shape)

# img, label = valset[0]

# print(img.shape)
# print(label.shape)

# img, label = testset[0]

# print(img.shape)
# print(label.shape)

# i=0
# for img, label in train_loader:
#     print(f"batch: {i}")
#     print(img.shape)
#     print(label.shape)
#     i+=1
    