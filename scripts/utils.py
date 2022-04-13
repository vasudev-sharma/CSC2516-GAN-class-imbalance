import argparse
import numpy as np
import torch
from torchvision import models
import scipy
import torch
import wandb
import torch.nn as nn
import os
import scipy.linalg
from torch.distributions import MultivariateNormal
import numpy as np
import random
import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Early stopping: Adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

def get_deterministic_run(SEED=None):
        # ADD SEED for consistent result
    SEED = 42
    np.random.seed(SEED)
    wandb.login()
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False
    os.environ['PYTHONHASHSEED'] = str(SEED)

    return

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    
def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, save_models=None, gen=None, disc=None,gen_pretrained_path=None,disc_pretrained_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_models = save_models
        self.gen  = gen
        self.disc  = disc
        self.gen_pretrained_path  = gen_pretrained_path
        self.disc_pretrained_path  = disc_pretrained_path
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.save_models(self.gen, self.disc, self.gen_pretrained_path, self.disc_pretrained_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_inception_model(path=None, device=device):
    if not path:
        model = models.inception_v3(pretrained=True)
    else:
        model = models.inception_v3(pretrained=False)
        model.load_state_dict(torch.load('/root/CSC2516-GAN-class-imbalance/inception_v3_google-1a9a5a14.pth'))
    # model.Conv2d_1a_3x3.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model = model.eval()

    return model

def matrix_sqrt(x):
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)

    return torch.tensor(y, device=x.device)

def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    '''
    Funciton for returning frechet distance between two Multivariate Normal distributions
    Parameters:
        mu_x: Mean of the first distribution
        my_x: Mean of the second distribution
        sigma_x: Covariance of the first distribution
        sigma_y: Covariance of the second distribution
    '''

    fid = (mu_x - mu_y).dot(mu_x-mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2*torch.trace(matrix_sqrt(sigma_x @ sigma_y))
    return fid



def get_input_dimensions(z_dim, input_shape, num_classes):
    generator_input_dim = z_dim + num_classes
    critic_input_dim = input_shape[0] + num_classes

    return generator_input_dim, critic_input_dim


# UNIT TEST
'''
mean1 = torch.Tensor([0, 0]) # Center the mean at the origin
covariance1 = torch.Tensor( # This matrix shows independence - there are only non-zero values on the diagonal
    [[1, 0],
     [0, 1]]
)
dist1 = MultivariateNormal(mean1, covariance1)

mean2 = torch.Tensor([0, 0]) # Center the mean at the origin
covariance2 = torch.Tensor( # This matrix shows dependence 
    [[2, -1],
     [-1, 2]]
)
dist2 = MultivariateNormal(mean2, covariance2)

assert torch.isclose(
    frechet_distance(
        dist1.mean, dist2.mean,
        dist1.covariance_matrix, dist2.covariance_matrix
    ),
    4 - 2 * torch.sqrt(torch.tensor(3.))
)

assert (frechet_distance(
        dist1.mean, dist1.mean,
        dist1.covariance_matrix, dist1.covariance_matrix
    ).item() == 0)

print("Success!")
'''
def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))


def preprocess(img):
    return torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)




def compute_FID(real_features_list, fake_features_list):
    # Concatenate the features
    real_features_all = torch.cat(real_features_list)
    fake_features_all = torch.cat(fake_features_list)

    # Mean and covariance
    mu_fake = fake_features_all.mean(0)
    mu_real = real_features_all.mean(0)
    sigma_fake = get_covariance(fake_features_all)
    sigma_real = get_covariance(real_features_all)

    # Print FID
    with torch.no_grad():
        print(f'The frechet_distance is:  {frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item()}')


def load_generator_and_discriminator(gen=None, disc=None, gen_pretrained_path='', disc_pretrained_path=''):
    if gen_pretrained_path == '':
        gen_pretrained_path = os.path.join(os.getcwd(), 'gen.pth')
        disc_pretrained_path = os.path.join(os.getcwd(), 'disc.pth')
    if gen is not None and disc is None:
        gen.load_state_dict(torch.load(gen_pretrained_path))
        print()
        print('Generator has been loaded')
        print()
        return gen
    elif disc is not None and gen is None:
        disc.load_state_dict(torch.load(disc_pretrained_path))
        print()
        print('Discriminator has been loaded')
        print()
        return disc
    elif not disc and not gen:
        disc.load_state_dict(torch.load(disc_pretrained_path))
        gen.load_state_dict(torch.load(gen_pretrained_path))
        print()
        print('Generator and Discriminator have been loaded')
        print()
        return disc, gen
    else:
        raise Exception("Generator and/or Discriminator are invalid")



def update_parser(parser):
    parser.add_argument('--batch_size', help="Batch Size", type=int, required=False, default=32)
    parser.add_argument('--lr', help="Learning Rate", type=float, required=False, default=1e-3)
    parser.add_argument('--epochs', help="Total Number of Epochs", type=int, required=False, default=30)
    parser.add_argument('--z_dim', help="Noise vector dimension", type=int, required=False, default=64)
    parser.add_argument('--display_step', help="Display steps for logging", type=int, required=False, default=200)
    parser.add_argument('--patience', help="Early stopping epochs", type=int, required=False, default=30)
    return parser

def save_models(gen=None, disc=None, gen_pretrained_path='', disc_pretrained_path=''):
    if gen_pretrained_path == '':
        gen_pretrained_path = os.path.join(os.getcwd(), 'gen.pth')
        disc_pretrained_path = os.path.join(os.getcwd(), 'disc.pth')
    if gen is not None and disc is None:
        torch.save(gen.state_dict(), gen_pretrained_path)
        wandb.save(gen_pretrained_path)

        # return 
    elif disc is not None and gen is None:
        torch.save(disc.state_dict(), disc_pretrained_path)
        wandb.save(disc_pretrained_path)
        # return 
    elif disc is not None and  gen is not None:
        torch.save(gen.state_dict(), gen_pretrained_path)
        torch.save(disc.state_dict(), disc_pretrained_path)

        wandb.save(gen_pretrained_path)
        wandb.save(disc_pretrained_path)
        # return
    elif disc is None and gen is None:
        raise Exception("Generator and/or Discriminator are invalid")

    print('...'*32)
    print(f"Models Generator and discriminator have been saved {gen_pretrained_path} and {disc_pretrained_path} respectively")
    print('...'*32)




def gan_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'RSNA, COVID', type=str, required=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    pass