import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, n_epochs, label_smoothing=0.0):
        self.model=model
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.device=device
        self.n_epochs=n_epochs
        self.label_smoothing=label_smoothing
    def train_epoch(self, epoch):
        self.model.train()
        loss_value=0.0
        for data, target in tqdm(self.train_loader, desc=f'Training @ epoch {epoch}'):
            data=data.to(self.device)
            target=target.to(self.device)
            self.optimizer.zero_grad()
            ans=self.model(data)
            loss=F.cross_entropy(ans,target, label_smoothing=self.label_smoothing)
            loss_value+=loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss_value/len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        correct=0
        total=0
        loss_sum=0.0
        for data, target in tqdm(self.val_loader, desc='Evaluating'):
            data=data.to(self.device)
            target=target.to(self.device)
            with torch.no_grad():
                ans=self.model(data)
                loss=F.cross_entropy(ans,target, reduction='sum')
                loss_sum+=loss.item()
            correct+=(ans.argmax(dim=1)==target).sum().item()
            total+=len(target)

        acc=correct/total if total>0 else 0.0
        val_loss=loss_sum/total if total>0 else 0.0
        return acc, val_loss

    def train(self):
        self.model=self.model.to(self.device)
        for epoch in range(self.n_epochs):
            train_loss=self.train_epoch(epoch)
            acc, val_loss=self.evaluate()
            print(f'Epoch {epoch} - TrainingLoss: {train_loss} ValLoss: {val_loss} - ValAccuracy: {acc}')
        return self.model