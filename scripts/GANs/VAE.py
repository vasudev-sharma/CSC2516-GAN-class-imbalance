
import  matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The encoder will output two things: Mean and Covariance matrix of Multivariate Normal Distribution where all the latent dimmensions are independent of each other
class Encoder(nn.Module):
    def __init__(self, im_channel=1, output_channel=32, hidden_dim=16):
        self.z_dim = output_channel
        self.encoder = nn.Sequential(
            self.make_disc_block(im_channel, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim*2),
            self.make_disc_block(hidden_dim*2, self.z_dim * 2, final_layer=True),
        )
        
    def make_disc_block(self, input_channels, output_channel, kernel_size, stride, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channel, kernel_size, stride),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channel, kernel_size, stride)
            )
    def forward(self, image):
        disc_pred = self.encoder(image)
        disc_pred = disc_pred.view(len(disc_pred), 1)

        # Since std is treated as a log for numerical stability, we take exponential on top of it.
        return disc_pred[:, :self.z_dim], disc_pred[:, self.z_dim:].exp() # Mean and std respectively


class Decoder(nn.Module):
    # The input is only z_dim and not z_dim because we are sampling from the probability distribution
    def __init__(self, z_dim=32, im_channel=1, hidden_dim=16):
        self.z_dim = z_dim
        self.decoder = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim*4, hidden_dim*2),
            self.make_gen_block(hidden_dim*2, hidden_dim),
            self.make_gen_block(hidden_dim, im_channel, final_layer=True)
        )

    def make_gen_block(input_channels, output_channels, kernel_size, stride, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Sigmoid()
            )
        

    def forward(self, noise):
        # Noise vector of dimmension: (batch_size, z_dim)
        noise = noise.view(len(noise), self.z_dim, 1, 1)
        return self.decoder(noise)

reconstruction_loss = nn.BCELoss(reduction='sum')

def kl_divergence_loss(q_dist):
    return kl_divergence(
        q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.std))
    ).sum(-1)

class VAE(nn.Module):
    def __init__(self, z_dim=32, im_channel=1, hidden_dim=64):
        self.z_dim = self.z_dim
        self.encoder = Encoder(im_channel, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, im_channel, hidden_dim)

    def forward(self, images):
        # Sample the image
        dist_mean, dist_cov = self.encoder(images) 
        q_dist = Normal(dist_mean, dist_cov) # `q_dist` is Multivariate Normal Distribution
        # Since q_dist is maximizing ELBO, therefore it is loose optimization and hence the resulting fake images are lower in fidelity
        noise = q_dist.rsample()  # rsample() to backpropagte 

        fake_images = self.decoder(noise)

        return fake_images, q_dist 

def show_tensor_images(img_tensor, num_images=25, img_size=(1, 28, 28)):
    images = img_tensor.detach().cpu()
    img_grid = make_grid(images[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(img_grid.permute(1, 2, 0).squeeze(), cmap='gray')


if __name__ == "__main__":
    transforms = transforms.Compose([
                    transforms.ToTensor()
    ])
    train_ds = datasets.MNIST(root='.', train=True, transforms=transforms)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=1024)
    


    vae = VAE.to(device)
    vae_opt = torch.optim.Adam(vae.parameters, lr=0.001)

    # Training Loop
    for epoch in range(10):
        print(f"This is {epoch + 1}")

        for idx, (images, _) in enumerate(enumerate(train_dl)):
            images = images.to(device)
            vae_opt.zero_grad()

            # Forward pass
            fake_images, q_dist = vae(images)

            # Compute loss
            loss = kl_divergence_loss(q_dist).sum() + reconstruction_loss(images, fake_images) # Compute the loss
            
            # Backward pass
            loss.backward()
            vae_opt.step() # Update gradients
        
        # Plotting fake and real images
        plt.subplot(1, 2, 1)
        plt.title(f'Epoch {epoch+1}: True images')
        show_tensor_images(images)

        plt.subplot(1, 2, 2)
        plt.title(f'Epoch {epoch+1}: Fake images')
        show_tensor_images(images)

        plt.savefig("Epoch: {epoch + 1}.png")












