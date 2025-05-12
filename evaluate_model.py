import yaml
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from datasets.covid19 import get_covid19_dataloaders, get_covid_loader
from models.vit import create_model
from trainers.utils import extract_features
from utils.metrics import evaluate_retrieval
from checkpoints.checkpoint import load_checkpoint


if __name__ == '__main__':
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # 1. Load model & optimizer
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg['model']['name'], cfg['model']['num_classes'], cfg['model']['pretrained']).to(device)

    # 2. Load checkpoint
    checkpoint_path = cfg['model']['checkpoint']['path']
    load_checkpoint(model, filename=checkpoint_path)

    # 3. Load data loader
    train_loader = get_covid_loader(cfg['dataset']['train_root'], cfg['train']['batch_size'])
    val_loader = get_covid_loader(cfg['dataset']['validation_root'], cfg['train']['batch_size'])

    # 4. Trích xuất đặc trưng
    features_train, labels_train = extract_features(model, train_loader, device)
    features, labels = extract_features(model, val_loader, device)

    # 5. Đánh giá
    metrics = evaluate_retrieval(features, labels, features_train, labels_train)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
