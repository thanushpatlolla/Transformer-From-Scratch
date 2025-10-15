import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm

class ImageDataLoader:
    def __init__(self, batch_size, dataset_name="cifar100", normalize=True, augmentation=False):
        self.batch_size=batch_size
        self.dataset_name=dataset_name
        self.normalize=normalize
        self.augmentation=augmentation
    def get_data(self):
        img_transform=[]        
        
        if(self.augmentation):
            aug_transform=[
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),  # or AutoAugment(CIFAR10)
            ]
            img_transform.extend(aug_transform)
            
        
        img_transform.append(transforms.ToTensor())
        
        if(self.normalize):
            if(self.dataset_name=="cifar100"):
                MEAN = [0.4914, 0.4822, 0.4465]
                STD = [0.2470, 0.2435, 0.2616]
                img_transform.append(transforms.Normalize(MEAN, STD))
            else:
                raise ValueError(f"Invalid dataset name: {self.dataset_name}")
            
            
        if(self.augmentation):
            img_transform.append(transforms.RandomErasing(p=0.25))
            
        if(self.dataset_name=="cifar100"):
            train_dataset=datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose(img_transform))
            val_dataset=datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose(img_transform))
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        train_loader=data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader=data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader