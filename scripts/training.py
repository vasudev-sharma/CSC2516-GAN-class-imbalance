from distutils.dir_util import copy_tree
import shutil
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func
import torchxrayvision as xrv


from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
# import random
# import logging
# import pandas as pd
import pickle
from .utils import EarlyStopping


use_gpu = torch.cuda.is_available()


def load_data(path, dataset_size=None, with_gan=False, data_aug=False, dataset="RSNA", im_channel=1, gan_data_path=''):
    # add data augmentations transforms here
    # TRAIN_WITH_GAN_FILENAME = "train_preprocessed_subset_{}_with_gan.csv".format(dataset_size)
    # TRAIN_WITHOUT_GAN_FILENAME = "train_preprocessed_subset_{}.csv".format(dataset_size)

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    # replace the paths for the dataset here
    # train_filename = TRAIN_WITH_GAN_FILENAME if with_gan else TRAIN_WITHOUT_GAN_FILENAME

    # COVID train
    if dataset == 'COVID':
        train_filename = os.path.join(os.getcwd(), 'data/covid-chestxray-dataset/metadata.csv')
    elif dataset == "RSNA":
        train_filename = os.path.join(os.getcwd(), 'data/RSNA_Pneumonia/stage_2_train_labels.csv')
    if data_aug:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224),
                                                transforms.ToTensor(),
                                                transforms.Lambda(lambda t: torch.permute(t, (1, 0, 2)))])

        data_aug_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip()])
        if dataset == "COVID":
            ds_covid = xrv.datasets.COVID19_Dataset(imgpath=path,
                                        csvpath=train_filename, transform=transform, data_aug=data_aug_transforms)
        elif dataset == "RSNA":
            ds_covid = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=path,
                                       csvpath=train_filename, transform=transform, extension='.dcm', data_aug=data_aug_transforms)
        elif dataset == "COVID-small":
            # TODO: Have same transforms as COVID-Xray Transforms
            transform = torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                                transforms.CenterCrop(224),
                                                transforms.Resize(224),
                                                transforms.ToTensor(),
                                                transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip()])
            ds_covid = ImageFolder(path, transform=transform, target_transform=lambda t: F.one_hot(torch.tensor(t), num_classes=3).float())
    else:
        
        if dataset == "COVID":
            ds_covid = xrv.datasets.COVID19_Dataset(imgpath=path,
                                        csvpath=train_filename, transform=transform)
        elif dataset == "RSNA":
            ds_covid = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=path,
                                    csvpath=train_filename, transform=transform, extension='.dcm')
        elif dataset == "COVID-small":
            # TODO: Have same transforms
            # Better performance is without data aug --> need to check why
            transform = torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1),
                                                transforms.CenterCrop(224),
                                                transforms.Resize(224),
                                                transforms.ToTensor()])
            ds_covid = ImageFolder(path, transform=transform, target_transform=lambda t: F.one_hot(torch.tensor(t), num_classes=3).float())
    

    if with_gan:
        # transform_gan = torchvision.transforms.Compose([
        #                                         transforms.Resize((64, 64)),            
        #                                         transforms.ToPILImage()
        #                             # transforms.Grayscale(num_output_channels=im_channel), # for FID
                        
        #                             ])
        if dataset == "COVID":
            ds_covid_gan = xrv.datasets.COVID19_Dataset(imgpath=path,
                                        csvpath=train_filename, transform=transform_gan)
        elif dataset == "RSNA":
            # ds_covid_gan = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=path,
            #                         csvpath=train_filename, transform=transform_gan, extension='.dcm')
            
            transform_gan = transforms.Compose([
                                    # transforms.Resize(299),
                                    # transforms.CenterCrop(299),
                                    transforms.Resize((64, 64)),            
                                    transforms.Grayscale(num_output_channels=im_channel), # for FID
                                    transforms.ToTensor(),
                                    transforms.Normalize(tuple([0.5] * im_channel), tuple([0.5] * im_channel)),
                                ])

            path = '/w/247/vasu/CSC2516-GAN-class-imbalance/data/RSNA_Pneumonia/preprocessed_data' # 62 % of the complete data
            ds_covid_gan = ImageFolder(path, transform=transform_gan, target_transform=lambda t: F.one_hot(torch.tensor(t), num_classes=2).float())
            print(ds_covid_gan.class_to_idx)
            print(len(ds_covid_gan))



            len_train = int(0.8 * (len(ds_covid_gan))) # Since the training images is 60% of the data, 0.2 is essentially 10% of the data. 0.8 is 50% of the data

            # 13375 = 50% of data
            # 26750 = 100% of data
            # Preprocess: Opacity 3740, Pneumonia: 12750: 16490
            print(len_train)
            len_test = len(ds_covid_gan) - len_train
            print(len_test)
            ds_covid_gan, _ = random_split(ds_covid_gan, [len_train, len_test])  


            
        elif dataset == "COVID-small":
       
            transform_gan = transforms.Compose([
                                    # transforms.Resize(299),
                                    # transforms.CenterCrop(299),
                                    transforms.Resize((64, 64)),            
                                    transforms.Grayscale(num_output_channels=im_channel), # for FID
                                    transforms.ToTensor(),
                                    transforms.Normalize(tuple([0.5] * im_channel), tuple([0.5] * im_channel)),
                                ])
            ds_covid_gan = ImageFolder(path, transform=transform_gan)
            print("**"*12)
            print(ds_covid_gan.class_to_idx)
            print("**"*12)
        if not gan_data_path:
            return ds_covid_gan, transform_gan
        else:
            
            print("*********Training uisng generated data***********")
            new_data_path = os.path.join(os.path.dirname(path), 'gan_original_images')
            try:
                os.path.exists(new_data_path)
            except:
                print("*************Deleting the original Directory**************")
                shutil.rmtree(new_data_path)

            # Make temp new generated directory: Original + GAN generated images
            # os.mkdir(new_data_path)

            # TODO: Remove hardcoding
            try:
                
                shutil.copytree(path, new_data_path)
                print("********Direcotry created************")
            except:
                print("******* Couldn't copy the file *********** ")

            try:
                copy_tree(gan_data_path, os.path.join(new_data_path, 'Covid-19'))
                print("*************Dataset has been processed************")
            except:
                print("Hello")
                print("******* Couldn't copy the file *********** ")

            ds_covid = ImageFolder(new_data_path, transform=transform_gan, target_transform=lambda t: F.one_hot(torch.tensor(t), num_classes=3).float())
            
            ###########
            # RSNA
            ###########
            # ds_covid_generated = ImageFolder("/w/247/vasu/CSC2516-GAN-class-imbalance/models/RSNA/DCGAN/rsna_50", transform=transform_gan, target_transform=lambda t: F.one_hot(torch.tensor(t), num_classes=2).float())
            # ds_covid = ds_covid_generated + ds_covid_gan

            # print("Length of new covid:", len(ds_covid_gan))

    len_train = int(0.8 * (len(ds_covid)))
    len_test = len(ds_covid) - len_train
    ds_covid_train, ds_covid_test = random_split(ds_covid, [len_train, len_test])  


    return ds_covid_train, ds_covid_test

def get_model(num_classes):
    model = xrv.models.DenseNet(num_classes=num_classes)
    print(model.classifier)
    return model


def preprocess_data(dataset):
    for idx, data in enumerate(dataset):
        data['lab']=np.nan_to_num(data['lab'],0)
        data['lab']=np.where(data['lab']==-1, 1, data['lab'])
    return dataset


def training(model, num_epochs, model_path, model_name, train_loader, valid_loader,lr=0.001, optimizer="momentum", dataset='COVID'):
    print("training")
    # hyperparameters
    criterion = nn.BCEWithLogitsLoss()
    if optimizer == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise Exception("Invalid optimizer")

    best_valid_loss = 10000
    PATH = model_path + model_name

    # going through epochs
    best_epoch = 0

    losses = {"val": [], "train": []}
    
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epochs):
        # training loss
        print("epoch", epoch)
        model.train()
        model.to("cuda:0")
        train_loss = 0
        count = 0
        for idx, data_all in enumerate(tqdm(train_loader)):
            if dataset == "COVID-small" or dataset == 'RSNA':
                data, target = data_all[0],  data_all[1]
            else:
                data, target = data_all['img'],  data_all['lab']
                
            count += 1
            # if count % 100 == 0:
            #     print("Count {}".format(count))
            #     sys.stdout.flush()
            data = data
            target = target
            data = data.to("cuda:0")
            target = target.to("cuda:0")
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation loss
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data_all in tqdm(valid_loader):
                if dataset == "COVID-small" or dataset == 'RSNA':
                    data, target = data_all[0],  data_all[1]
                else:
                    data, target = data_all['img'],  data_all['lab']
                    
                data = data
                target = target
                data = data.to("cuda:0")
                target = target.to("cuda:0")
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)




        # Log onto wandb

        wandb.log({'Train Loss': train_loss,
                    'Val Loss': valid_loss})


        # Early Stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print()
            print("="*32 + "Early Stopping" + "="*32)

            break
        # saves best epoch
        print(f'Epoch: {epoch + 1}/{num_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')
        losses['val'].append(valid_loss)
        losses['train'].append(train_loss)
        if valid_loss < best_valid_loss:
            best_epoch = epoch + 1
            torch.save(model.state_dict(), PATH)
            best_valid_loss = valid_loss
        print("Best Valid Loss so far:", best_valid_loss)
        print("Best epoch so far: ", best_epoch)


    with open(model_path + "{}_losses.pkl".format(model_name), "wb") as handle:
        pickle.dump(losses, handle)

    return best_valid_loss, best_epoch


def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            # print(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            # TODO: Handle ROCAUC for only one class
            outAUROC.append(0)
            # pass
    return outAUROC


def testing(model, test_loader, nnClassCount, class_names, dataset="COVID"):
    if use_gpu:
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

    model.eval()
    # print("class count")
    # print(nnClassCount)
    # print(class_names)

    # print()
    # print("targets")
    with torch.no_grad():
        for batch_idx, data_all in enumerate(tqdm(test_loader)):
            if batch_idx % 100 == 0:
                print(batch_idx)

            if dataset == "COVID-small" or dataset == "RSNA":
                    data, target = data_all[0],  data_all[1]
            else:
                data, target = data_all['img'],  data_all['lab']

            data = data
            target = target
            # print(target.shape)
            # print(target)
            target = target.cuda()
            data = data.to("cuda:0")
            outGT = torch.cat((outGT, target), 0).cuda()

            # bs, c, h, w = data.size()
            # varInput = data.view(-1, c, h, w)

            out = model(data)
            outPRED = torch.cat((outPRED, out), 0)

    aurocIndividual = computeAUROC(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()

    for idx, class_auroc in enumerate(aurocIndividual):
        # class_name = f'test_aucroc_{idx}'
        wandb.config.update({f'test_aucroc_{idx}': class_auroc})
        # wandb.config.class_name = class_auroc
    wandb.config.test_aucroc = aurocMean 

    # print(len(aurocIndividual))
    # print(aurocIndividual)

    print('AUROC mean ', aurocMean)
    sys.stdout.flush()

    results = {}
    for i in range(0, len(aurocIndividual)):
        results[class_names[i]] = [aurocIndividual[i]]
        print(class_names[i], ' ', aurocIndividual[i])
    sys.stdout.flush()
    return results
    # results_df = pd.DataFrame(results)
    # results_df.to_csv(output_path + "auc_results_{}.csv".format(model_id), index=False)

    # return outGT, outPRED


if __name__ == '__main__':
    path_covid = '/root/CSC2516-GAN-class-imbalance/data/covid-chestxray-dataset/images'
    path_rsna = '/root/CSC2516-GAN-class-imbalance/data/RSNA_Pneumonia/stage_2_train_images'
    ds_covid_train, ds_covid_test = load_data(path=path_rsna)
    print(ds_covid_train[0])
