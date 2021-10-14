import pickle as pkl
import tarfile
import tarfile
from urllib.request import urlretrieve
import os


import numpy as np
import skimage
import skimage.io
import skimage.transform
import torchvision


def _compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)

    bg = background[x:x+dw, y:y+dh]
    return np.abs(bg - digit).astype(np.uint8)


def _mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def _create_mnistm(X, rand, background_data):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):

        if i % 10000 == 0:
            print('Processing example', i)

        bg_img = rand.choice(background_data)

        d = _mnist_to_img(X[i])
        d = _compose_image(d, bg_img)
        X_[i] = d

    return X_


def create_mnistm():
  if os.path.exists('mnistm_data.pkl'):
    return

  if not os.path.exists("BSR_bsds500.tgz"):
    urlretrieve("http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz", "BSR_bsds500.tgz")
  print('Creating MNIST-M... That may takes a minute')
  BST_PATH = 'BSR_bsds500.tgz'

  rand = np.random.RandomState(42)

  f = tarfile.open(BST_PATH)
  train_files = []
  for name in f.getnames():
      if name.startswith('BSR/BSDS500/data/images/train/'):
          train_files.append(name)

  print('Loading BSR training images')
  background_data = []
  for name in train_files:
      try:
          fp = f.extractfile(name)
          bg_img = skimage.io.imread(fp)
          background_data.append(bg_img)
      except:
          continue

  mnist_train = torchvision.datasets.MNIST('.', train=True, download=True)
  mnist_test = torchvision.datasets.MNIST('.', train=False, download=True)

  print('Building train set...')
  train = _create_mnistm(mnist_train.data.numpy(), rand, background_data)
  print('Building test set...')
  test = _create_mnistm(mnist_test.data.numpy(), rand, background_data)

  # Save dataset as pickle
  with open('mnistm_data.pkl', 'wb+') as f:
      pkl.dump({ 'x_train': train, 'x_test': test, "y_train": mnist_train.targets.numpy(), "y_test": mnist_test.targets.numpy()}, f, pkl.HIGHEST_PROTOCOL)

  print("Done!")
