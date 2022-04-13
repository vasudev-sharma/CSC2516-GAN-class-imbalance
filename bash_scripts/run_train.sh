#!/bin/bash

# Command to Train
# srun --partition smallgpunodes --nodelist gpunode27 -c 2 --gres=gpu:1 --mem=15000M bash_scripts/run_train.sh 

# python scripts/GANs/DCGAN_GP_conditional.py --dataset MNIST --user vasu --with_gan True
# python scripts/GANs/FID.py --dataset MNIST --user vasu --with_gan True
# python scripts/GANs/DCGAN_GP_conditional.py --dataset MNIST --user vasu --with_gan True

## MNIST 

# DCGAN
python scripts/GANs/DCGAN.py --dataset MNIST --user vasu --with_gan True --epochs 200 --lr 2e-3 --display_step 400 --z_dim 128 --GAN_type DCGAN --batch_size 64 --im_channel 3 --patience 30 --n_class_generate 1 --num_images_per_class 20

# DCGAN_GP
# python scripts/GANs/DCGAN_GP.py --dataset MNIST --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 200 --z_dim 128 --GAN_type DCGAN_GP --batch_size 64 --im_channel 3 --patience 30

# SNGAN
# python scripts/GANs/SNGAN.py --dataset MNIST --user vasu --with_gan True --epochs 100 --lr 2e-4 --display_step 200 --z_dim 128 --GAN_type SNGAN --batch_size 64 --im_channel 3 --patience 30

# LSGAN
# python scripts/GANs/LSGAN.py --dataset MNIST --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 200 --z_dim 128 --GAN_type LSGAN --batch_size 64 --im_channel 3 --patience 30

# DCGAN_GP_conditional
# python scripts/GANs/DCGAN_GP_conditional.py --dataset MNIST --user vasu --with_gan True --epochs 2 --lr 2e-4 --display_step 400 --z_dim 128 --batch_size 64 --im_channel 3 --GAN_type DCGAN_GP_conditional --n_class_generate 1 --num_images_per_class 20 --patience 30 --critic_repeats 5 --c_lambda 10

## COVID -19

# DCGAN
# python scripts/GANs/DCGAN.py --dataset COVID-small --user vasu --with_gan True --epochs 200 --lr 2e-3 --display_step 50 --z_dim 128 --GAN_type DCGAN --batch_size 32  --im_channel 3 --patience 30

# DCGAN_GP
# python scripts/GANs/DCGAN_GP.py --dataset COVID-small --user vasu --with_gan True --epochs 200 --lr 2e-3 --display_step 100 --z_dim 128 --GAN_type DCGAN_GP --batch_size 32  --im_channel 3  --patience 30

# DCGAN_GP_conditional
# python scripts/GANs/DCGAN_GP_conditional.py --dataset COVID-small --user vasu --with_gan True --epochs 100 --lr 2e-4 --display_step 200 --z_dim 128 --GAN_type DCGAN_GP_conditional --batch_size 32  --im_channel 3 --n_class_generate 1 --num_images_per_class 20 --patience 30 --critic_repeats 2 --c_lambda 5

# SNGAN
# python scripts/GANs/SNGAN.py --dataset COVID-small --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 100 --z_dim 128 --GAN_type SNGAN --batch_size 32  --im_channel 3 --patience 30

# LSGAN
# python scripts/GANs/LSGAN.py --dataset COVID-small --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 100 --z_dim 128 --GAN_type LSGAN --batch_size 32 --im_channel 3 --patience 30



# RSNA dataset

# DCGAN
# python scripts/GANs/DCGAN.py --dataset RSNA --user vasu --with_gan True --epochs 200 --lr 2e-3 --display_step 50 --z_dim 128 --GAN_type DCGAN --batch_size 32  --im_channel 3 --patience 30

# DCGAN_GP
# python scripts/GANs/DCGAN_GP.py --dataset RSNA --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 100 --z_dim 128 --GAN_type DCGAN --batch_size 32  --im_channel 3 --patience 30

# DCGAN_GP_conditional
# python scripts/GANs/DCGAN_GP_conditional.py --dataset RSNA --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 100 --z_dim 128 --GAN_type DCGAN_GP_conditional --batch_size 32  --im_channel 3 --patience 30 --critic_repeats 5 --c_lambda 10

# SNGAN
# python scripts/GANs/SNGAN.py --dataset RSNA --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 100 --z_dim 128 --GAN_type SNGAN --batch_size 32  --im_channel 3 --patience 30

# LSGAN
# python scripts/GANs/LSGAN.py --dataset RSNA --user vasu --with_gan True --epochs 200 --lr 2e-4 --display_step 100 --z_dim 128 --GAN_type LSGAN --batch_size 32 --im_channel 3 --patience 30






# python test.py