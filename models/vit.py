import timm
import torch.nn as nn

def create_model(name='vit_base_patch16_224', num_classes=10, pretrained=True):
    if name == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    elif name == 'vit_small_patch16_224':
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name")

    model.head = nn.Linear(model.head.in_features, num_classes)
    return model