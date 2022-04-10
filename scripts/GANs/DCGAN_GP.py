import wandb 
import torch
import argparse
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch.nn as nn
from torchvision import transforms, datasets
from main import get_paths
from scripts.training import load_data
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from scripts.utils import save_models
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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




class Critic(nn.Module):

    def __init__(self, im_channel=1, hidden_dim=16):
        super(Critic, self).__init__()

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
Test your make_disc_block() function
'''
num_test = 100

'''gen = Generator()
disc = Discriminator()
test_images = gen(get_noise(num_test, gen.z_dim))

# Test the hidden block
test_hidden_block = disc.make_disc_block(1, 5, kernel_size=6, stride=3)
hidden_output = test_hidden_block(test_images)

# Test the final block
test_final_block = disc.make_disc_block(1, 10, kernel_size=2, stride=5, final_layer=True)
final_output = test_final_block(test_images)

# Test the whole thing:
disc_output = disc(test_images)


# Test the hidden block
assert tuple(hidden_output.shape) == (num_test, 5, 8, 8)
# Because of the LeakyReLU slope
assert -hidden_output.min() / hidden_output.max() > 0.15
assert -hidden_output.min() / hidden_output.max() < 0.25
assert hidden_output.std() > 0.5
assert hidden_output.std() < 1

# Test the final block

assert tuple(final_output.shape) == (num_test, 10, 6, 6)
assert final_output.max() > 1.0
assert final_output.min() < -1.0
assert final_output.std() > 0.3
assert final_output.std() < 0.6

# Test the whole thing:

assert tuple(disc_output.shape) == (num_test, 1)
assert disc_output.std() > 0.25
assert disc_output.std() < 0.5
print("Success!")
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

    gradient = torch.autograd.grad(inputs=mixed_images, outputs=mixed_scores, grad_outputs=torch.ones_like(mixed_scores)
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


def get_gen_loss(critic_fake_predictions):
    gen_loss = -1 * torch.mean(critic_fake_predictions)
    return gen_loss

def get_critic_loss(critic_fake_predictions, critic_real_predictions, gp, c_lambda):

    critic_loss = torch.mean(critic_fake_predictions) - torch.mean(critic_real_predictions) + c_lambda * gp

    return critic_loss


def test_get_gradient(image_shape):
    real = torch.randn(*image_shape, device=device) + 1
    fake = torch.randn(*image_shape, device=device) - 1
    epsilon_shape = [1 for _ in image_shape]
    epsilon_shape[0] = image_shape[0]
    epsilon = torch.rand(epsilon_shape, device=device).requires_grad_()
    gradient = get_gradient(critic, real, fake, epsilon)
    assert tuple(gradient.shape) == image_shape
    assert gradient.max() > 0
    assert gradient.min() < 0
    return gradient


def test_gradient_penalty(image_shape):
    bad_gradient = torch.zeros(*image_shape)
    bad_gradient_penalty = gradient_penalty(bad_gradient)
    assert torch.isclose(bad_gradient_penalty, torch.tensor(1.))

    image_size = torch.prod(torch.Tensor(image_shape[1:]))
    good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
    good_gradient_penalty = gradient_penalty(good_gradient)
    assert torch.isclose(good_gradient_penalty, torch.tensor(0.))

    random_gradient = test_get_gradient(image_shape)
    random_gradient_penalty = gradient_penalty(random_gradient)
    assert torch.abs(random_gradient_penalty - 1) < 0.1

# test_gradient_penalty((256, 1, 28, 28)): Assertion Error is encountered: check it

def get_one_hot_labels(labels, classes):
    return F.one_hot(labels, num_classes=classes)

def combine_vectors(x, y):
    ''' Function for combining One hot encoded vector with the noise vectors
    '''

    return torch.cat([x.float(), y.float()], dim=1)




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
    parser.add_argument('--GAN_type', help = 'DCGAN, DCGAN_GP, SNGAN, DCGAN_GP_cond', type=str, required=False)
    parser.add_argument('--user', type=str, required=True)

    args = parser.parse_args()


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
    gradient = test_get_gradient((256, 1, 28, 28))
    print("Success!")
    print("Success!")


    # Hyperparameters and loss
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 200
    z_dim = 64
    display_step = 200
    lr = 2e-4
    device = 'cuda'
    batch_size = 64


    beta1 = 0.5
    beta2 = 0.999

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
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
    wandb.config.dataset = args.dataset
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
    critic = Critic().to(device)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))


    combined = combine_vectors(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]));
    # Check exact order of elements
    assert torch.all(combined == torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]]))
    # Tests that items are of float type
    assert (type(combined[0][0].item()) == float)
    # Check shapes
    combined = combine_vectors(torch.randn(1, 4, 5), torch.randn(1, 8, 5));
    assert tuple(combined.shape) == (1, 12, 5)
    assert tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12).long()).shape) == (1, 30, 12)
    print("Success!")


    gen = gen.apply(weights_init)
    critic = critic.apply(weights_init)

    # num_epochs = 50
    mean_critic_loss = 0.0 
    mean_generator_loss = 0.0 
    curr_step = 0


    critic_repeats = 5
    c_lambda = 10
    critic_losses = []
    generator_losses = []


    for epoch in tqdm(range(num_epochs)):
        
        for real, _ in tqdm(dataloader):
            curr_batch_size = len(real)
            real = real.to(device)

            mean_critic_loss = 0.0
            for _ in range(critic_repeats):

                # Update Critic
                critic_opt.zero_grad()
                fake_noise = get_noise(curr_batch_size, z_dim, device=device)
                fake_images = gen(fake_noise)
                critic_fake_preds = critic(fake_images.detach())
                critic_real_preds = critic(real)
                

                epsilon = torch.randn(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(critic, real, fake_images.detach(), epsilon)
                gp = gradient_penalty(gradient)
                critic_loss = get_critic_loss(critic_fake_preds, critic_real_preds, gp, c_lambda)

                mean_critic_loss += critic_loss.item() / critic_repeats

                # Update graidents
                critic_loss.backward(retain_graph=True)
                critic_opt.step()
            critic_losses += [mean_critic_loss]



            # Update Generator
            gen_opt.zero_grad()

            fake_noise = get_noise(curr_batch_size, z_dim, device=device)
            fake_images = gen(fake_noise)
            fake_predictions = critic(fake_images)

            gen_loss = get_gen_loss(fake_predictions)

            gen_loss.backward()
            gen_opt.step()

            generator_losses += [gen_loss.item()]


            # Log into wandb
            wandb.log({
                "epoch": epoch,
                "Generator Loss": sum(generator_losses) / len(generator_losses),
                "Discriminator Loss": sum(critic_losses) / len(critic_losses)
            })

            # Visualization code

            if curr_step > 0 and curr_step % display_step == 0:
                print(f'Step: {curr_step} | Generator Loss:{sum(generator_losses[-display_step:]) / display_step} | Discriminator Loss: {sum(critic_losses[-display_step:]) / display_step}')
                # noise_vectors = get_noise(curr_batch_size, z_dim, device=device)
                # fake_images = gen(noise_vectors)
                show_tensor_images(fake_images, type="fake")
                show_tensor_images(real, type="real")

                # mean_generator_loss = 0.0
                # mean_discriminator_loss = 0.0

            curr_step += 1

    print("Training is Completed ")    


    #################################
    # Save Models and log onto wandb
    #################################

    save_models(gen=gen, disc=critic)