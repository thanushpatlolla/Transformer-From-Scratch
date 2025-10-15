import torch
from transformer import Transformer
from image_dataloader import ImageDataLoader
from training import Trainer
from tokenizers import ImageTokenizer
from heads import ViTHead
import torch.optim as optim

def cifar100(device, n_epochs=100):
    d_model=384
    tokenizer=ImageTokenizer(img_size=32, patch_size=4, in_channels=3, d_model=d_model)
    head=ViTHead(d_model=d_model, n_classes=100, norm="rms")
    model=Transformer(tokenizer=tokenizer, head=head, n_layers=8, d_model=d_model, n_head=12, d_head=32, d_v=32, n_linear=2, d_linear=d_model*4, act="gelu", norm="rms", normpos="pre")

    loader=ImageDataLoader(batch_size=128, dataset_name="cifar100", normalize=True, augmentation=True)
    train_loader, val_loader=loader.get_data()
    optimizer=optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.05)

    steps_per_epoch = len(train_loader)
    warmup_steps = int(0.05 * steps_per_epoch * n_epochs)
    total_steps  = steps_per_epoch * n_epochs

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        ],
        milestones=[warmup_steps]
    )

    trainer=Trainer(model, train_loader, val_loader, optimizer, scheduler, device=device, n_epochs=n_epochs, label_smoothing=0.1)
    trainer.train()
    return model
