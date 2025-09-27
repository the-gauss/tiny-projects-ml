import torch, torchvision
import json
import os

DATASET_PATH = 'data/images'

with open('data/cat_to_name.json', 'r') as f:
    CATEGORY_NAMES = json.load(f)

class OxfordFlowers(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.__data_path = data_path
        self.__split = split
        self.img_and_labels = self.__create_labels()    # (image, label) list of tuples
        self.transform = transform

    # Create a list of tuples (image, label) from file
    def __create_labels(self):
        split_folder = os.path.join(self.__data_path, self.__split)   # 'data/images/train' etc.
        labels = sorted(os.listdir(split_folder))
        label_to_idx = {name: idx for idx, name in enumerate(labels)}   # {"1":0, "2":1...}

        img_and_labels = []
        for label in labels:
            label_folder = os.path.join(split_folder, label)
            for file in os.listdir(label_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filename = os.path.join(label_folder, file)
                    img_and_labels.append((filename,label_to_idx[label]))
        return img_and_labels

    def __len__(self):
        return len(self.img_and_labels)

    def __getitem__(self, idx):
        filename, label = self.img_and_labels[idx]
        image = torchvision.io.read_image(filename).float()/255.0   # Convert from PyTorch int [0,255] to float [0,1] because Normalize() expects this
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataset(split: str ='train', additional_transforms: list =None):
    transforms_list = [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ]

    if additional_transforms:
        transforms_list.extend(additional_transforms)

    transforms = torchvision.transforms.Compose(transforms_list)

    dataset = OxfordFlowers(DATASET_PATH, split=split, transform=transforms)
    return dataset

def create_dataloader(split: str = 'train', additional_transforms: list = None, **kwargs):
    dataset = create_dataset(split, additional_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
    return dataloader