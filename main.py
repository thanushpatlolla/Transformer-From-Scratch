import torch
from transformer import Transformer
from image_dataloader import ImageDataLoader
from training import Trainer
from tokenizers import ImageTokenizer
from heads import ViTHead
import torch.optim as optim
from defaults import cifar100

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

model=cifar100(device)