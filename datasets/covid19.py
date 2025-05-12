import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_covid19_dataloaders(data_dir='./raw', batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Tách train/test (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader

def get_covid_loader(data_dir='./data/dataset/train', batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader