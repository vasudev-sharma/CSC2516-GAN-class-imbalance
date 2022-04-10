import argparse
import wandb 
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), type='fake'):
    ''' Function for visualizing images
    '''

    img_unflat = image_tensor.cpu().view(-1, *size)
    img_grid = make_grid(img_unflat[:num_images], nrow=5)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze().numpy(), cmap='gray')
    if type == 'fake':
        images = wandb.Image(img_grid.permute(1, 2, 0).squeeze().numpy(), caption="Fake Images")
        wandb.log({"Fake Image": images})
    elif type == 'real':
        images = wandb.Image(img_grid.permute(1, 2, 0).squeeze().numpy(), caption="Real Images")
        wandb.log({"Real Image": images})
    else:
        raise Exception("Invalid Type entered: Real / Fake")
    plt.show()


### Generator Block
def generator_block(input_dim, output_dim):

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


def test_generator_block(input_dim, output_dim):
    gen = generator_block(input_dim, output_dim)
    assert len(gen) == 3

    assert type(gen[0]) == nn.Linear
    assert type(gen[1]) == nn.BatchNorm1d
    assert type(gen[2]) == nn.ReLU

    print(gen)



class Generator(nn.Module):

    def __init__(self, z_dim, hidden_dim=128, input_dim=784):
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


def get_noise(num_samples, z_dim, device='cpu'):

    return torch.randn((num_samples, z_dim), device=device)




## Discrimator

def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Discriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.sel

        self.disc = nn.Sequential(
            get_discriminator_block(input_dim, hidden_dim*4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),

            nn.Linear(hidden_dim, 1), # Real or Fake
        )


    def forward(self, inputs):
        return self.disc(inputs)



def get_disc_loss(gen, disc, z_dim, criterion, real_samples, num_images, device='cuda'):
    # Forward pass
    noise_vectors = get_noise(num_images, z_dim, device=device)

    fake_samples = gen(noise_vectors)
    fake_predictions = disc(fake_samples.detach())
    fake_targets = torch.zeros_like(fake_predictions)
    fake_loss = criterion(fake_predictions, fake_targets)


    real_predictions = disc(real_samples)
    real_targets = torch.ones_like(real_predictions)
    real_loss = criterion(real_predictions, real_targets)

    disc_loss = (fake_loss + real_loss) / 2
    return disc_loss



# def test_disc_loss(num_images=10):
#     z_dim = 64

#     gen = Generator(z_dim).to(device)

#     disc = Discriminator().to(device)

#     get_disc_loss(gen, disc, z_dim, )


def get_gen_loss(gen, disc, z_dim, criterion, num_images, device='cuda'):
    # pass
    noise_vectors = get_noise(num_images, z_dim, device=device)
    fake_samples = gen(noise_vectors)
    fake_predictions = disc(fake_samples)

    gen_loss = criterion(fake_predictions, torch.ones_like(fake_predictions))

    return gen_loss


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()


    parser.add_argument('--with_gan', type=bool, default=True, required=False)
    parser.add_argument('--dataset', help = 'RSNA, COVID, COVID-small, MNIST', type=str, default="MNIST", required=False)
    parser.add_argument('--user', type=str, required=True)

    args = parser.parse_args()

    ## Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)

    test_generator_block(3, 5)


    # Hyperparameters and loss
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 200
    z_dim = 64
    display_step = 500
    lr = 1e-5
    device = 'cuda'
    batch_size = 128

    wandb.init(entity='vs74', project='GAN')
    config = { 'num_epochs' : num_epochs,
    'z_dim' : z_dim,
    'display_step' : display_step,
    'lr' : lr,
    'device' : device,
    'batch_size' : batch_size,
        }

    wandb.config.update(config)

    # Dataloader
    dataloader = DataLoader(datasets.MNIST('.', download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    # Initialize the networks
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)



    mean_generator_loss = 0.0
    mean_discriminator_loss = 0.0
    curr_step = 0 

    for epoch in tqdm(range(num_epochs)):
        
        for real, _ in tqdm(dataloader):
            current_batch_size = real.size(0)

            # Flatten the image
            real = real.view(current_batch_size, -1).to(device)

            ## Update the discriminator first
            disc_opt.zero_grad()

            disc_loss = get_disc_loss(gen, disc, z_dim, criterion, real, current_batch_size, device=device)

            # Discriminator Loss
            disc_loss.backward(retain_graph=True)

            disc_opt.step()


            ## Update the Discriminator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, z_dim, criterion, current_batch_size, device='cuda')
            gen_loss.backward()
            gen_opt.step()


            ## Keep track of Discriminator and Generator Loss
            mean_generator_loss += gen_loss.item() / display_step

            mean_discriminator_loss += disc_loss.item() / display_step

            wandb.log({
                "epoch": epoch,
                "Generator Loss": mean_generator_loss,
                "Discriminator Loss": mean_discriminator_loss
            })

            if curr_step > 0 and curr_step % display_step == 0:
                print(f'Step: {curr_step} | Generator Loss:{mean_generator_loss} | Discriminator Loss: {mean_discriminator_loss}')
                noise_vectors = get_noise(current_batch_size, z_dim, device=device)
                fake_images = gen(noise_vectors)
                show_tensor_images(fake_images, type="fake")
                show_tensor_images(real, type="real")
                mean_generator_loss = 0
                mean_discriminator_loss = 0

            curr_step += 1

    print("Training is Completed ")    








