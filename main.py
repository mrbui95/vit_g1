import yaml
import torch
from models.vit import create_model
from datasets.covid19 import get_covid19_dataloaders, get_covid_loader
from trainers.trainer import train_one_epoch, train_with_contrastive_loss
from torch.optim import Adam

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg['model']['name'], cfg['model']['num_classes'], cfg['model']['pretrained']).to(device)

    #train_loader, test_loader = get_covid19_dataloaders(cfg['dataset']['root'], cfg['train']['batch_size'])
    train_loader = get_covid_loader(cfg['dataset']['train_root'], cfg['train']['batch_size'])
    optimizer = Adam(model.parameters(), lr=float(cfg['train']['lr']))

    checkpoint_path = cfg['model']['checkpoint']['path']

    for epoch in range(cfg['train']['epochs']):
        loss = train_with_contrastive_loss(model, train_loader, optimizer, device, epoch, checkpoint_path)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")