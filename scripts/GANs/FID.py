import torch
from tqdm.auto import tqdm
import torch.nn as nn
from scripts.utils import get_inception_model, preprocess, get_covariance, frechet_distance, load_generator_and_discriminator
from torchvision import transforms
from scripts.GANs.DCGAN_GP_conditional import Generator, get_input_dimensions
from torchvision import datasets
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_dim = 64
batch_size = 128

generator_dim, critic_dim = get_input_dimensions(z_dim, input_shape=(1, 28, 28), num_classes=10)

# Load pretrained GAN
gen = Generator(generator_dim).to(device)
gen = load_generator_and_discriminator(gen=gen)
gen = gen.eval()

inception_model = get_inception_model(device=device)


# Transformation for MNIST

transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


dataloader = DataLoader(datasets.MNIST('.', download=True, transform=transform), batch_size=batch_size, shuffle=True)

fake_features_list = []
real_features_list = []
with torch.no_grad():
    try:
        for real, _ in tqdm(dataloader):
            real_samples = real.to(device)

            real_features = inception_model(real_samples).detach().to()

    except:
        print("Error in the loop")

# # Compute FID after training
# compute_FID(real_features_list, fake_features_list)


