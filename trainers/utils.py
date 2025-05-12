import torch

def extract_features(model, dataloader, device):
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = model.forward_features(imgs)

            # Nếu mô hình ViT: feats shape có thể là (B, 197, D), dùng [CLS] token
            if len(feats.shape) == 3:
                feats = feats[:, 0]  # dùng CLS token

            all_feats.append(feats)
            all_labels.append(labels)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels
