import torch

def save_checkpoint(model, optimizer, epoch, loss, filename='./checkpoints/checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"✅ Checkpoint saved at '{filename}'")


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth'):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    print(f"✅ Checkpoint loaded from '{filename}' at epoch {epoch}")
    return epoch, loss
