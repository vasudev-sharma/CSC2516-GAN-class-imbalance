import numpy as np
import torch
from torchvision import models
import scipy
import torch
import wandb
import os
import scipy.linalg
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Early stopping: Adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
    model.Conv2d_1a_3x3.conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
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

# UNIT TEST

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
    if gen is not None and disc is None:
        gen.load_state_dict(torch.load(gen_pretrained_path))
        return gen
    elif disc is not None and gen is None:
        disc.load_state_dict(torch.load(disc_pretrained_path))
        return disc
    elif not disc and not gen:
        disc.load_state_dict(torch.load(disc_pretrained_path))
        gen.load_state_dict(torch.load(gen_pretrained_path))
        return disc, gen
    else:
        raise Exception("Generator and/or Discriminator are invalid")

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
