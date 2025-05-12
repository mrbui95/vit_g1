import torch
import torch.nn.functional as F
from tqdm import tqdm
from checkpoints.checkpoint import save_checkpoint
from .loss import contrastive_koleo_loss, contrastive_koleo_loss_cosin

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    l = total_loss / len(dataloader)
    save_checkpoint(model, optimizer, epoch, l, './checkpoints/checkpoint_cross_entropy.pth')
    return l

def train_with_contrastive_loss(model, dataloader, optimizer, device, epoch, check_point_path):
    index = 0
    for imgs, labels in dataloader:
        index += 1
        # print("imgs shape: " + str(imgs.shape) + ", label shape: " + str(labels.shape))
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        feats = model.forward_features(imgs)
        loss, l_contr_val, l_koleo_val = contrastive_koleo_loss_cosin(feats, labels)
        if (index % 50 == 0):
            print(f"Total {index}: {loss.item():.4f}, Contrastive: {l_contr_val:.4f}, KoLeo: {l_koleo_val:.4f}")
        
        loss.backward()
        optimizer.step()
    print(f"Total {index}: {loss.item():.4f}, Contrastive: {l_contr_val:.4f}, KoLeo: {l_koleo_val:.4f}")

    save_checkpoint(model, optimizer, epoch, loss, check_point_path)
    return loss