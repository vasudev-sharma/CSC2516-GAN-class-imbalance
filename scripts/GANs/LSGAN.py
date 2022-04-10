import wandb 
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from scripts.utils import save_models
from scripts.training import load_data
from main import get_paths


# wandb.login(key=['202040aaac395bbf5a4a47d433a5335b74b7fb0e'])

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), type='fake'):
    ''' Function for visualizing images
    '''
    image_tensor = (image_tensor + 1) / 2
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



class Generator(nn.Module):
    def __init__(self, z_dim=10, im_channel=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(

            self.make_gen_block(self.z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_channel, kernel_size=4, final_layer=True)
        )
    
    def make_gen_block(self, input_channels, output_channels, kernel_size=3,  stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
                nn.Tanh() # Tanh Activation is used here
            )

    def forward(self, noise_vectors):
        z = self.unsqueeze_noise(noise_vectors)
        return self.gen(z)

    def unsqueeze_noise(self, noise_vectors):
        return noise_vectors.view(noise_vectors.size(0), self.z_dim, 1, 1)

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

'''
## Test Discriminator
gen = Generator()
num_test = 100

# Test the hidden block
test_hidden_noise = get_noise(num_test, gen.z_dim)
test_hidden_block = gen.make_gen_block(10, 20, kernel_size=4, stride=1)
test_uns_noise = gen.unsqueeze_noise(test_hidden_noise)
hidden_output = test_hidden_block(test_uns_noise)

# Check that it works with other strides
test_hidden_block_stride = gen.make_gen_block(20, 20, kernel_size=4, stride=2)

test_final_noise = get_noise(num_test, gen.z_dim) * 20
test_final_block = gen.make_gen_block(10, 20, final_layer=True)
test_final_uns_noise = gen.unsqueeze_noise(test_final_noise)
final_output = test_final_block(test_final_uns_noise)

# Test the whole thing:
test_gen_noise = get_noise(num_test, gen.z_dim)
test_uns_gen_noise = gen.unsqueeze_noise(test_gen_noise)
gen_output = gen(test_uns_gen_noise)

# UNIT TESTS
assert tuple(hidden_output.shape) == (num_test, 20, 4, 4)
assert hidden_output.max() > 1
assert hidden_output.min() == 0
assert hidden_output.std() > 0.2
assert hidden_output.std() < 1
assert hidden_output.std() > 0.5

assert tuple(test_hidden_block_stride(hidden_output).shape) == (num_test, 20, 10, 10)

assert final_output.max().item() == 1
assert final_output.min().item() == -1

assert tuple(gen_output.shape) == (num_test, 1, 28, 28)
assert gen_output.std() > 0.5
assert gen_output.std() < 0.8
print("Success!")


'''


class Discriminator(nn.Module):

    def __init__(self, im_channel=1, hidden_dim=16):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            self.make_disc_block(im_channel, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim*2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True)
        )
    
    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Conv2d(input_channels, output_channels, kernel_size, stride)
    
    def forward(self, inputs):
        disc_pred = self.disc(inputs)
        return disc_pred.view(len(disc_pred), -1)

'''
# Test your make_disc_block() function

num_test = 100

gen = Generator(im_channel=3)
disc = Discriminator(im_channel=3)
test_images = gen(get_noise(num_test, gen.z_dim))

# Test the hidden block
test_hidden_block = disc.make_disc_block(1, 5, kernel_size=6, stride=3)
# hidden_output = test_hidden_block(test_images)

# Test the final block
# test_final_block = disc.make_disc_block(1, 10, kernel_size=2, stride=5, final_layer=True)
# final_output = test_final_block(test_images)

# Test the whole thing:
# disc_output = disc(test_images)


# # Test the hidden block
# assert tuple(hidden_output.shape) == (num_test, 5, 8, 8)
# # Because of the LeakyReLU slope
# assert -hidden_output.min() / hidden_output.max() > 0.15
# assert -hidden_output.min() / hidden_output.max() < 0.25
# assert hidden_output.std() > 0.5
# assert hidden_output.std() < 1

# # Test the final block

# assert tuple(final_output.shape) == (num_test, 10, 6, 6)
# assert final_output.max() > 1.0
# assert final_output.min() < -1.0
# assert final_output.std() > 0.3
# assert final_output.std() < 0.6

# # Test the whole thing:

# assert tuple(disc_output.shape) == (num_test, 1)
# assert disc_output.std() > 0.25
# assert disc_output.std() < 0.5
# print("Success!")

'''

def get_gradient(critic, real, fake, epsilon):
    """ Returns the critic scores with respect to the mixes of fake and real images
    Parameters:
        critic: The critic model
        real: batch of real images
        fake: batch of fake images
        epsilon: vector of uniform random proportion of real/fake per mixed image
    Returns:
        gradient: gradient of the critic's score with respect to the mixed image
    """

    # Mix images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Critic scores on the mixed images
    mixed_scores = critic(mixed_images)

    gradient = torch.autograd.grad(inputs=mixed_images, ouputs=mixed_scores, grad_outputs=torch.ones_like(mixed_scores)
    , create_graph=True, retain_graph=True)[0]

    return gradient


def gradient_penalty(gradient):
    '''
    Returns the gradient penalty given the graident
    Parameters:
        graident: the gradient of the cirtic's scores with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradient
    gradient = gradient.view(len(gradient), -1)

    # Compute the norm
    gradient_norm = gradient.norm(2, dim=1)

    # Penalty term
    penalty = torch.mean((gradient_norm - 1)**2)

    return penalty



# Weights initializations: with mean and std 0 and 2 respectively
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--with_gan', type=bool, default=True, required=False)
    parser.add_argument('--dataset', help = 'RSNA, COVID, COVID-small, MNIST', type=str, default="MNIST", required=False)
    parser.add_argument('--user', type=str, required=True)

    args = parser.parse_args()



        
    # Hyperparameters and loss
    criterion = nn.MSELoss()
    num_epochs = 200
    z_dim = 64
    display_step = 500
    lr = 2e-4
    device = 'cuda'
    batch_size = 128


    beta1 = 0.5
    beta2 = 0.999


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    wandb.init(entity='vs74', project='GAN')
    config = { 'num_epochs' : num_epochs,
    'z_dim' : z_dim,
    'display_step' : display_step,
    'lr' : lr,
    'device' : device,
    'batch_size' : batch_size,
        }
    wandb.config.update(config)


    num_classes = {"MNIST": 10,
                    "COVID": 25,
                    "COVID-small": 3,
                    "RSNA": 2}

    if args.dataset == "MNIST":

        transform = transforms.Compose([
            # transforms.Resize(299),
            # transforms.CenterCrop(299),
            transforms.Grayscale(num_output_channels=1), # for FID
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
        dataloader = DataLoader(datasets.MNIST('.', download=True, transform=transform), batch_size=batch_size, shuffle=True)
        # generator_dim, critic_dim = get_input_dimensions(z_dim, input_shape=(3, 28, 28), num_classes=10)

    elif args.dataset == "COVID" or args.dataset == "COVID-small" or args.dataset == "RSNA":
        data_path, output_path, model_path = get_paths(args, args.user)
        ds, transform = load_data(data_path, dataset_size=None, with_gan=args.with_gan, data_aug=False, dataset=args.dataset)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # TODO: Play with different size of the generated image
        # generator_dim, critic_dim = get_input_dimensions(z_dim, input_shape=(3, 28, 28), num_classes=num_classes[args.dataset])

    else:
        raise Exception("Invalid Dataset Entered")



        

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

    
    
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # num_epochs = 50
    mean_discriminator_loss = 0.0 
    mean_generator_loss = 0.0 
    curr_step = 0


    for epoch in tqdm(range(num_epochs)):
        
        for real, _ in tqdm(dataloader):
            curr_batch_size = len(real)
            real = real.to(device)

            # Update Discriminator
            disc_opt.zero_grad()
            fake_noise = get_noise(curr_batch_size, z_dim, device=device)
            fake_images = gen(fake_noise)
            fake_predictions = disc(fake_images.detach())
            disc_fake_loss = criterion(fake_predictions, torch.zeros_like(fake_predictions))
            real_predictions = disc(real)
            disc_real_loss = criterion(real_predictions, torch.ones_like(real_predictions))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            mean_discriminator_loss += disc_loss.item() / display_step
            disc_loss.backward(retain_graph=True)
            disc_opt.step()


            # Update Generator
            gen_opt.zero_grad()
            fake_noise = get_noise(curr_batch_size, z_dim, device=device)
            fake_images = gen(fake_noise)
            fake_predictions = disc(fake_images)
            gen_loss = criterion(fake_predictions, torch.ones_like(fake_predictions))

            gen_loss.backward()
            gen_opt.step()

            mean_generator_loss += gen_loss.item() / display_step


            # Log into wandb
            wandb.log({
                "epoch": epoch,
                "Generator Loss": mean_generator_loss,
                "Discriminator Loss": mean_discriminator_loss
            })

            # Visualization code

            if curr_step > 0 and curr_step % display_step == 0:
                print(f'Step: {curr_step} | Generator Loss:{mean_generator_loss} | Discriminator Loss: {mean_discriminator_loss}')
                # noise_vectors = get_noise(curr_batch_size, z_dim, device=device)
                # fake_images = gen(noise_vectors)
                show_tensor_images(fake_images, type="fake", size=(1, 28, 28))
                show_tensor_images(real, type="real", size=(1, 28, 28))
                mean_generator_loss = 0
                mean_discriminator_loss = 0

            curr_step += 1

    print("Training is Completed ")    


    #################################
    # Save Models and log onto wandb
    #################################

    save_models(gen=gen, disc=disc)


