from ast import arguments
import os
import wandb 
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from scripts.utils import InceptionV3, calculate_fretchet, EarlyStopping, save_models, get_deterministic_run, update_parser, get_input_dimensions, load_generator_and_discriminator
from scripts.GANs.DCGAN_GP_conditional import get_one_hot_labels, combine_vectors, interpolation_noise
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

        if args.dataset == "MNIST":
            self.gen = nn.Sequential(

                self.make_gen_block(self.z_dim, hidden_dim * 4),
                self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
                self.make_gen_block(hidden_dim * 2, hidden_dim),
                self.make_gen_block(hidden_dim, im_channel, kernel_size=4, final_layer=True)
            )

        else:

             self.gen = nn.Sequential(

                self.make_gen_block(self.z_dim, hidden_dim * 8, 4, 1, 0),
                # self.make_gen_block(hidden_dim * 16, hidden_dim * 8, 4, 2, 1),
                self.make_gen_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
                self.make_gen_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
                self.make_gen_block(hidden_dim * 2, hidden_dim, 4, 2, 1),
                self.make_gen_block(hidden_dim, im_channel, 4, 2, 1, final_layer=True)
            )

    
    def make_gen_block(self, input_channels, output_channels, kernel_size=3,  stride=2, padding=0, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Tanh() # Tanh Activation is used here
            )

    def forward(self, noise_vectors):
        z = self.unsqueeze_noise(noise_vectors)
        # for layer in self.gen(z):
        #     print(layer.shape)
        return self.gen(z)

    def unsqueeze_noise(self, noise_vectors):
        return noise_vectors.view(noise_vectors.size(0), self.z_dim, 1, 1)

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

class Discriminator(nn.Module):

    def __init__(self, im_channel=1, hidden_dim=64):
        super(Discriminator, self).__init__()

        if args.dataset == "MNIST":

            self.disc = nn.Sequential(
                self.make_disc_block(im_channel, hidden_dim),
                self.make_disc_block(hidden_dim, hidden_dim*2),
                self.make_disc_block(hidden_dim * 2, 1, final_layer=True)
            )
        else:
            self.disc = nn.Sequential(
                self.make_disc_block(im_channel, hidden_dim, 4, 2, 1),
                self.make_disc_block(hidden_dim, hidden_dim*2, 4, 2, 1),
                self.make_disc_block(hidden_dim*2, hidden_dim*4, 4, 2, 1),
                self.make_disc_block(hidden_dim*4, hidden_dim*8, 4, 2, 1),
                self.make_disc_block(hidden_dim*8, 1, 4, 1, 0, final_layer=True),
                # self.make_disc_block(hidden_dim*16, 1, 4, 1, 0, final_layer=True)
            )
    
    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
    
    def forward(self, inputs):
        disc_pred = self.disc(inputs)
        return disc_pred.view(len(disc_pred), -1)

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

    get_deterministic_run()

    parser = argparse.ArgumentParser()


    parser.add_argument('--with_gan', type=bool, default=True, required=False)
    parser.add_argument('--dataset', help = 'RSNA, COVID, COVID-small, MNIST', type=str, default="MNIST", required=False)
    parser.add_argument('--GAN_type', help = 'DCGAN, DCGAN_GP, LSGAN, SNGAN, DCGAN_GP_conditional', type=str, required=False)
    parser.add_argument('--user', type=str, required=True)
    parser.add_argument('--im_channel', type=int, required=False, default=1)
    parser.add_argument('--n_class_generate', type=int, required=False, default=1)
    parser.add_argument('--num_images_per_class', type=int, required=False, default=20)
    parser = update_parser(parser)

    args = parser.parse_args()


    # Hyperparameters and loss
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = args.epochs
    z_dim = args.z_dim
    lr = args.lr
    device = device

    '''# Chest X ray params
    display_step = 100
    batch_size = 32
'''
    # MNIST  params
    display_step = args.display_step
    batch_size = args.batch_size
    # print("Hello World")

    im_channel = args.im_channel

    beta1 = 0.5
    beta2 = 0.999



    # wandb.login()
    wandb.init(entity='vs74', project='GAN')
    wandb.config.update(args)


    num_classes = {"MNIST": 10,
                    "COVID": 25,
                    "COVID-small": 3,
                    "RSNA": 2}

    if args.dataset == "MNIST":

        transform = transforms.Compose([
            # transforms.Resize(299),
            # transforms.CenterCrop(299),
            transforms.Grayscale(num_output_channels=im_channel), # for FID
            transforms.ToTensor(),
            transforms.Normalize(tuple([0.5] * im_channel), tuple([0.5] * im_channel)),
        ])
        dataloader = DataLoader(datasets.MNIST('.', download=True, transform=transform), batch_size=batch_size, shuffle=True)
        # generator_dim, critic_dim = get_input_dimensions(z_dim, input_shape=(3, 28, 28), num_classes=10)

        generator_dim, critic_dim = get_input_dimensions(z_dim, input_shape=(im_channel, 28, 28), num_classes=num_classes[args.dataset])


        model_path = os.path.join(os.getcwd(), "models")

    elif args.dataset == "COVID" or args.dataset == "COVID-small" or args.dataset == "RSNA":
        data_path, output_path, model_path = get_paths(args, args.user)
        ds, transform = load_data(data_path, dataset_size=None, with_gan=args.with_gan, data_aug=False, dataset=args.dataset, im_channel=args.im_channel)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)


        generator_dim, critic_dim = get_input_dimensions(z_dim, input_shape=(im_channel, 64, 64), num_classes=num_classes[args.dataset])

        # TODO: Play with different size of the generated image
        # generator_dim, critic_dim = get_input_dimensions(z_dim, input_shape=(3, 28, 28), num_classes=num_classes[args.dataset])

    else:
        raise Exception("Invalid Dataset Entered")


    gen = Generator(generator_dim, im_channel=im_channel).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    disc = Discriminator(critic_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)


    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model=model.cuda()


    # Use early stopping for FID
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_models=save_models, gen=gen, disc=disc, gen_pretrained_path=os.path.join(model_path, 'gen.pth'), disc_pretrained_path=os.path.join(model_path, 'disc.pth'))

    #################################
    # Save Models and log onto wandb
    #################################

    model_path = os.path.join(model_path, args.dataset, os.path.basename(__file__)[:-3])
    print(model_path)
    # save_models(gen=gen, disc=critic, gen_pretrained_path=os.path.join(model_path, 'gen.pth'), disc_pretrained_path=os.path.join(model_path, 'disc.pth'))


    # num_epochs = 50
    mean_discriminator_loss = 0.0 
    mean_generator_loss = 0.0 
    curr_step = 0



    for epoch in tqdm(range(num_epochs)):
        
        for real, labels in tqdm(dataloader):
            curr_batch_size = len(real)
            real = real.to(device)
            labels = labels.to(device)

            # Update Discriminator
            disc_opt.zero_grad()
            fake_noise = get_noise(curr_batch_size, z_dim, device=device)

            # Conditional GAN on labels
            labels_one_hot = get_one_hot_labels(labels, classes=num_classes[args.dataset])
            image_one_hot_labels = labels_one_hot[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, real.size(2), real.size(3))

            fake_noise_combined = combine_vectors(fake_noise, labels_one_hot)


            fake_images = gen(fake_noise_combined)

            fake_images_and_labels = combine_vectors(fake_images, image_one_hot_labels)
            real_images_and_labels = combine_vectors(real, image_one_hot_labels)

            fake_predictions = disc(fake_images_and_labels.detach())
            disc_fake_loss = criterion(fake_predictions, torch.zeros_like(fake_predictions))
            real_predictions = disc(real_images_and_labels)
            disc_real_loss = criterion(real_predictions, torch.ones_like(real_predictions))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            mean_discriminator_loss += disc_loss.item() / display_step
            disc_loss.backward(retain_graph=True)
            disc_opt.step()


            # Update Generator
            gen_opt.zero_grad()
            fake_noise = get_noise(curr_batch_size, z_dim, device=device)

            # Conditional Labels
            labels_one_hot = get_one_hot_labels(labels, classes=num_classes[args.dataset])
            image_one_hot_labels = labels_one_hot[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, real.size(2), real.size(3))

            fake_noise_combined = combine_vectors(fake_noise, labels_one_hot)


            fake_images = gen(fake_noise_combined)

            fake_images_and_labels = combine_vectors(fake_images, image_one_hot_labels)

            fake_predictions = disc(fake_images_and_labels)

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
                if args.dataset == "MNIST":
                    show_tensor_images(fake_images, type="fake", size=(im_channel, 28, 28))
                    show_tensor_images(real, type="real", size=(im_channel, 28, 28))
                else:
                    show_tensor_images(fake_images, type="fake", size=(im_channel, 64, 64))
                    show_tensor_images(real, type="real", size=(im_channel, 64, 64))
                mean_generator_loss = 0
                mean_discriminator_loss = 0

            curr_step += 1

        fretchet_dist=calculate_fretchet(real, fake_images, model) 
        wandb.log({'FID': fretchet_dist})

        early_stopping(fretchet_dist, model)

        if early_stopping.early_stop:
            print()
            print("="*32 + "Early Stopping" + "="*32)
            break

    print("Training is Completed ")    


    if not os.path.exists(os.path.join(model_path, 'gen.pth')):
        # Save the model
        print()
        print("*******************Best FID checkpointing didn't work**********")
        print()
        save_models(gen=gen, disc=disc, gen_pretrained_path=os.path.join(model_path, 'gen.pth'), disc_pretrained_path=os.path.join(model_path, 'disc.pth'))



    ##############################
    # Classes to Generate
    ##############################

    # Load the best FID score of Generator
    gen = load_generator_and_discriminator(gen=gen, gen_pretrained_path=os.path.join(model_path, 'gen.pth'))

    gen = gen.eval()

    n_interpolation = args.num_images_per_class # num of classes you want to generate

    n_class_generate = args.n_class_generate

    interpolation_label = get_one_hot_labels(torch.Tensor([n_class_generate]).long(), num_classes[args.dataset]).repeat(n_interpolation, 1).float()


    noise = get_noise(n_interpolation, z_dim, device=device)
    noise_and_labels = combine_vectors(noise, interpolation_label.to(device))
    fake = gen(noise_and_labels)



    # Save the images to generated directory
    if os.path.exists(os.path.join(model_path, 'generated')):
        shutil.rmtree(os.path.join(model_path, 'generated'))
        print("********Old Directory Deleted*********")
    
    os.mkdir(os.path.join(model_path, 'generated'))
    print("********New Directory Created*********")

    
    # Save the images in 'model_path / generated' directory
    for idx, img in tqdm(enumerate(fake)):
        img_path =  os.path.join(model_path, 'generated', str(idx))
        # img_pil = Image.fromarray(img.permute(1, 2, 0).detach().cpu().numpy(), mode='RGB').convert('L')
        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(img_path+'.png')

    print("The images have been generated in the directory:-- ", os.path.join(model_path, 'generated'))




    



