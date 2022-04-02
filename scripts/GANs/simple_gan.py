import torch
from tqdm.auto import tqdm
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

torch.manual_seed(42)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    ''' Function for visualizing images
    '''

