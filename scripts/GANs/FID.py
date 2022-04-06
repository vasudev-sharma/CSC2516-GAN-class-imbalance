import torch
import torch.nn as nn
from scripts.utils import get_inception_model, preprocess, get_covariance, frechet_distance
from torchvision import transforms
from scripts.GANs.DCGAN_GP_conditional import Generator, get_dimensions
from torchvision import datasets
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_dim = # TODO
# Load pretrained GAN
gen = Generator(z_dim).to(device)



# Transformation for MNIST

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


dataloader = DataLoader(datasets.MNIST('.', download=True, transform=transform), batch_size=batch_size, shuffle=True)


# # Compute FID after training
# compute_FID(real_features_list, fake_features_list)



