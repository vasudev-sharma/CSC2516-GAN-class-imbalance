import torch
from tqdm.auto import tqdm
import torch.nn as nn
from scripts.utils import get_inception_model, preprocess, get_covariance, frechet_distance, load_generator_and_discriminator, compute_FID
from torchvision import transforms
from scripts.GANs.DCGAN_GP_conditional import Generator, get_input_dimensions, get_noise, get_one_hot_labels, combine_vectors
from torchvision import datasets
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GAN_type', help = 'DCGAN, DCGAN_GP, SNGAN, DCGAN_GP_cond', type=str, required=False)
parser.add_argument('--dataset', help = 'RSNA, COVID, COVID-small, MNIST', type=str, required=False)
args = parser.parse_args()



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
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

if args.dataset == "MNIST":
    dataloader = DataLoader(datasets.MNIST('.', download=True, transform=transform), batch_size=batch_size, shuffle=True)

# TODO: Check if conditional noise vector and image is needed
fake_features_list = []
real_features_list = []
with torch.no_grad():
        for real, labels in tqdm(dataloader):
            real_samples = real.to(device)
            labels = labels.to(device)

            real_features = inception_model(real_samples).detach().to('cpu')
            real_features_list.append(real_features)

            fake_noise = get_noise(len(real), z_dim, device=device)

            # Conditional GAN on labels
            labels_one_hot = get_one_hot_labels(labels, classes=10)
            image_one_hot_labels = labels_one_hot[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, real.size(2), real.size(3))

            fake_noise_combined = combine_vectors(fake_noise, labels_one_hot)
            fake_samples = preprocess(gen(fake_noise_combined))

            fake_features = inception_model(fake_samples.to(device)).detach().to('cpu')
            fake_features_list.append(fake_features)



# # Compute FID after training
compute_FID(real_features_list, fake_features_list)


