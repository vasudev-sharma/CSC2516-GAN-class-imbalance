#!/bin/bash

# python scripts/GANs/DCGAN_GP_conditional.py --dataset MNIST --user vasu --with_gan True
# python scripts/GANs/FID.py --dataset MNIST --user vasu --with_gan True
# python scripts/GANs/DCGAN_GP_conditional.py --dataset MNIST --user vasu --with_gan True

## MNIST 

# DCGAN
# python scripts/GANs/DCGAN.py --dataset MNIST --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 200 --z_dim 64 --GAN_type DCGAN --batch_size 128 --im_channel 3 


## COVID -19

# DCGAN
python scripts/GANs/DCGAN.py --dataset COVID-small --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 50 --z_dim 64 --GAN_type DCGAN --batch_size 16  --im_channel 3


# DCGAN_GP

# python test.py