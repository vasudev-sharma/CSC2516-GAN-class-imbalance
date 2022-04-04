# Inspired from coursera

import torch
from tqdm.auto import tqdm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

torch.manual_seed(42)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    ''' Function for visualizing images
    '''

    img_unflat = image_tensor.cpu().view(-1, *size)
    img_grid = make_grid(img_unflat[:num_images], nrow=5)
    plt.imshow(img_grid.permute(1, 2, 0), cmap='gray')
    plt.show()


### Generator Block
def generator_block(input_dim, output_dim):

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(),
    )


def test_generator_block(input_dim, output_dim):
    gen = generator_block(input_dim, output_dim)
    assert len(gen) == 3

    assert type(gen[0]) == nn.Linear
    assert type(gen[1]) == nn.BatchNorm1d
    assert type(gen[2]) == nn.ReLU

    print(gen)

test_generator_block(3, 5)



class Generator(nn.Module):

    def __init__(self, z_dim, hidden_dim, input_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim 
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.gen = nn.Sequential(
            generator_block(z_dim, hidden_dim),
            generator_block(hidden_dim, hidden_dim * 2),
            generator_block(hidden_dim * 2, hidden_dim * 4),
            generator_block(hidden_dim * 4, hidden_dim * 8),

            # Final block
            nn.Linear(hidden_dim * 8, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, noise):
        return self.gen(noise)


def noise(num_samples, z_dim, device='cpu'):

    return torch.randn((num_samples, z_dim), device=device)




## Discrimator

def get_discriminator_block(input_dim, ouput_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.ouptu

        self.disc = nn.Sequential(
            get_discriminator_block(input_dim, hidden_dim*4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),

            nn.Linear(hidden_dim, 1) # Real or Fake
        )


    def forward(self, inputs):
        return self.disc(inputs)



## Training Loop