# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2, VGG16, imagenet_utils
from collections import defaultdict
from optparse import OptionParser


if __name__ == '__main__':
  argParser = OptionParser()

  argParser.add_option("--dataset", type="string", default="covid_small")
  argParser.add_option("--epochs", type="int", default=100)
  argParser.add_option("--model", type="string", default="VGG16")

  (options, args) = argParser.parse_args()

  dataset_name = options.dataset
  epochs = options.epochs
  model_type = options.model

  print("Generating tsne plot")
  print("Pretrained model {}".format(model_type))
  print("Dataset {}, epochs {}".format(dataset_name, epochs))
  print("=="*50)

  # %% --------------------------------------- Set-Up --------------------------------------------------------------------
  SEED = 42
  os.environ['PYTHONHASHSEED'] = str(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  # %% -------------------------------------- Data Prep ------------------------------------------------------------------
  save_file_path = ""
  gen_path = ""

  if dataset_name == "rsna_medium":
    x, y = np.load('x_train_rsna_medium.npy')[:1000], np.load('y_train_rsna_medium.npy')[:1000]
    x_test, y_test = np.load('x_val_rsna_medium.npy'), np.load('y_val_rsna_medium.npy')
    gen_path = 'RSNA_medium_{}_iter_bagan_gp_results/my_generator.h5'.format(epochs)
    save_file_path = 'RSNA_medium_{}_iter_bagan_gp_results/_{}_{}_epochs_{}_tsne.png'.format(epochs, 
        dataset_name, epochs, model_type)
  elif dataset_name == "rsna_small":
    x, y = np.load('x_train_rsna_small.npy')[:1000], np.load('y_train_rsna_small.npy')[:1000]
    x_test, y_test = np.load('x_val_rsna_small.npy'), np.load('y_val_rsna_small.npy')
    gen_path = 'RSNA_small_{}_iter_bagan_gp_results/my_generator.h5'.format(epochs)
    save_file_path = 'RSNA_small_{}_iter_bagan_gp_results/_{}_{}_epochs_{}_tsne.png'.format(epochs, 
        dataset_name, epochs, model_type)
  elif dataset_name == "covid_small":
    x, y = np.load('x_train_covid.npy')[:1000], np.load('y_train_covid.npy')[:1000]
    x_test, y_test = np.load('x_val_covid.npy'), np.load('y_val_covid.npy')
    gen_path = 'covid_{}_epochs_bagan_gp_results/my_generator.h5'.format(epochs)
    save_file_path = 'covid_{}_epochs_bagan_gp_results/_{}_{}_epochs_{}_tsne.png'.format(epochs, 
        dataset_name, epochs, model_type)

  n_classes = len(np.unique(y))
  inputShape = x[0].shape
  generator = load_model(gen_path)
  aug_size = 100

  for c in range(n_classes):
      sample_size = aug_size
      label = np.ones(sample_size) * c
      noise = np.random.normal(0, 1, (sample_size, generator.input_shape[0][1]))
      #print('Latent dimension:', generator.input_shape[0][1])
      gen_sample = generator.predict([noise, label])
      print("gen_sample", gen_sample.shape)
      gen_imgs = (gen_sample*0.5 + 0.5)*255
      x = np.append(x, gen_imgs, axis=0)
      y = np.append(y, label)
      print('Augmented dataset size:', sample_size, 'Total dataset size:', len(y))
  x_train, y_train = x, y

  preprocess = imagenet_utils.preprocess_input
  x_train = preprocess(x_train)
  x_test = preprocess(x_test)

  # %% -------------------------------------- Model Setup ----------------------------------------------------------------

  # Get the feature output from the pre-trained model ResNet50, VGG16, Inception, etc.
  
  if model_type == "VGG16":
    pretrained_model = VGG16(include_top=False, input_shape=inputShape, weights="imagenet")
  elif model_type == "ResNet50":
    pretrained_model = ResNet50(include_top=False, input_shape=inputShape, weights="imagenet")
  
  for layer in pretrained_model.layers:
      layer.trainable = False
  x = pretrained_model.layers[-1].output
  x = GlobalAveragePooling2D()(x)
  model = Model(pretrained_model.input, x)

  # %% -------------------------------------- TSNE Visualization ---------------------------------------------------------
  def tsne_plot(model, save_file_path=save_file_path):
    plt.figure(figsize=(8, 8))
    color = plt.get_cmap('tab10')

    latent = model.predict(x_train) # pretrained model receives the data
    tsne_model = TSNE(n_components=2, init='random', random_state=0) # tsne
    tsne_data = tsne_model.fit_transform(latent)

    x = []
    y = []
    for value in tsne_data:
        x.append(value[0])
        y.append(value[1])

    x = np.array(x)
    y = np.array(y)

    x_real = x[:-4 * aug_size]
    y_real = y[:-4 * aug_size]
    real_label = y_train[:-4 * aug_size]

    x_generated = x[-4 * aug_size:]
    y_generated = y[-4 * aug_size:]
    generated_label = y_train[-4 * aug_size:]

    loop = 0
    markers = ['o', 'x']
    color_label = ["green", "blue"]
    message_label = ["real", "fake"]

    plt.scatter(x_real, y_real, marker='o', c="red", label="real")
    plt.scatter(x_generated, y_generated, marker='x', c="blue", label="fake")
    plt.title("Improved BAGAN: T-sne for {}, {} dataset".format(model_type, dataset_name))
    plt.legend()
    plt.savefig(save_file_path)
    print("Figure saved: {}".format(save_file_path))
      
  tsne_plot(model)
