"""
Main training script

Functions
----------
load_data
read_img
read_csv
generator
build_model
main

Class
----------
ThreadsafeIterator
"""
import csv
import numpy as np
import os.path
import sklearn
import threading
import cv2

from keras import layers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


def load_data(file_path: str, ):
    """Read the csv and return X, y

    Parameters
    ----------
    file_path : str
        Path to CSV file

    Returns
    ----------
    X : array-like
    y : array-like
    """

    # center,left,right,steering,throttle,brake,speed
    data = read_csv(file_path)
    data_path = os.path.abspath(os.path.dirname(file_path))
    data_path = os.path.join(data_path, "IMG")

    img_list = []
    steering_list = []

    for center, left, right, steering, throttle, brake, speed in data:
        img_list.append(os.path.join(data_path, center.split("/")[-1]))
        img_list.append(os.path.join(data_path, left.split("/")[-1]))
        img_list.append(os.path.join(data_path, right.split("/")[-1]))

        correction = 0.25
        steering_list.append(steering)
        steering_list.append(steering + correction)
        steering_list.append(steering - correction)

    X = np.asarray(img_list)
    y = np.asarray(steering_list).astype(np.float32)

    return X, y


def read_csv(file_path):
    """Read a csv

    Parameters
    ----------
    file_path : str
        File path to csv

    Returns
    ----------
    data : list
        [row1, row2, ...]
        where row1 = [center, left, right, steering, throttle, brake, speed]
    """
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        return [[item.strip() if idx < 3 else float(item) for idx, item in enumerate(row)] for i, row in enumerate(reader) if i > 0]


class ThreadsafeIterator:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadsafeIterator(f(*a, **kw))
    return g


def crop_resize(img, new_size=(32, 32), crop=(50, 0, 20, 0)):
    """Crop and resize image

    Parameters
    ----------
    img : array-like, shape (H, W, C)
        Input image
    new_size : tuple (H, W)
    crop : tuple, shape (Top, Right, Bottom, Left)

    Returns
    ----------
    img : array-like
        Resized and cropped image
    """
    H, W, C = img.shape

    img = img[crop[0]:H - crop[2], crop[3]:W - crop[1], :]

    return cv2.resize(img, (new_size))


@threadsafe_generator
def generator(X_sample, y_sample, batch_size=32, train=True):
    """Dataset generator

    Parameters
    ----------
    X_sample : array-like, shape(N, ?, ?, ?)
        X images

    y_sample : array-like
        y labels. In this case, steering wheel angle

    batch_size : int (default: 32)
        Minibatch size

    train : bool (default: True)
        If it's True, no image augmentation is done

    Yields
    ----------
    X_batch : array-like, shape same as X_sample
    y_batch : array-like, shape same as y_sample
    """
    N = len(X_sample)

    while True:
        for i in range(0, N, batch_size):
            X_batch = X_sample[i:i + batch_size]
            y_batch = y_sample[i:i + batch_size]

            X_batch = np.asarray([crop_resize(cv2.imread(img_path)) for img_path in X_batch])

            if train:
                X_batch = np.concatenate((X_batch, [np.fliplr(img) for img in X_batch]))
                y_batch = np.concatenate((y_batch, -y_batch))
                X_batch = augment_image(X_batch)

            yield sklearn.utils.shuffle(X_batch, y_batch)


def augment_image(img_array):
    """Augment images randomly

    Parameters
    ----------
    img_array : array-like shape(N, W, H, C)

    Returns
    ----------
    img_array : array-like
        Augmented images
    """

    def st(aug):
        return iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([st(iaa.Multiply(mul=(0.25, 1.50))),  # Brightness
                          st(iaa.Crop(percent=(0, 0.1))),  # Random Crop
                          st(iaa.Dropout((0, 0.7))),  # Pixel Dropout
                          st(iaa.Invert(1)),
                          st(iaa.Affine(translate_percent=(0, 0.3),
                                        shear=(-45, 45))),  # Shear
                          st(iaa.Superpixels(p_replace=(0, 0.25),
                                             n_segments=100))],
                         random_order=True)

    # Visualize Augmentation
    # idx = np.random.choice(np.arange(len(img_array)))
    # seq.show_grid(img_array[idx], cols=8, rows=8)
    # assert 0

    return seq.augment_images(img_array)


def build_model(X):
    """Build a Keras model

    Parameters
    ----------
    X : layers.Input placeholder, shape(None, ?, ?, ?)
        Input image

    Returns
    ----------
    model : Model class
    """
    def samplewise_normalize(img):
        N, H, W, D = K.get_variable_shape(img)

        tmp = K.flatten(img)
        numerator = tmp - K.mean(tmp, 0)
        denominator = K.std(tmp, 0) + 1e-8

        normalized = numerator / denominator

        return K.reshape(normalized, (-1, H, W, D))

    X_process = X

    X_process = layers.Lambda(samplewise_normalize)(X_process)
    X_process = layers.Conv2D(3, (1, 1))(X_process)

    net = layers.Conv2D(24, (5, 5), strides=(2, 2))(X_process)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    net = layers.Conv2D(36, (5, 5), strides=(2, 2))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    net = layers.Conv2D(64, (3, 3))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    net = layers.Conv2D(64, (3, 3))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    net = layers.Flatten()(net)

    net = layers.Dense(100)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    net = layers.Dense(50)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    net = layers.Dense(10)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    net = layers.Dense(1)(net)

    model = Model(X, net)
    model.compile("adam", "mse")

    return model


def main(args):
    """Everything happens here

    Parameters
    ----------
    args : Parsed Arguments
    """
    X_train, y_train = load_data(args.path)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    print(X_train.shape)
    print(X_val.shape)

    train_gen = generator(X_train, y_train, batch_size=args.batch_size)
    val_gen = generator(X_val, y_val, batch_size=args.batch_size, train=False)

    input_shape = next(train_gen)[0].shape[1:]
    X = layers.Input(shape=input_shape)
    model = build_model(X)

    if os.path.isfile(args.model):
        print("Loading weights")
        model = load_model(args.model)

    else:
        print("No model is found")

    ckpt = ModelCheckpoint(filepath=args.model,
                           verbose=1,
                           mode='min',
                           save_best_only=True)
    early_stop = EarlyStopping(patience=3, mode='min')

    model.fit_generator(train_gen,
                        steps_per_epoch=X_train.shape[0] / args.batch_size,
                        callbacks=[ckpt, early_stop],
                        validation_data=val_gen,
                        validation_steps=X_val.shape[0] / args.batch_size,
                        max_q_size=args.batch_size * 100,
                        workers=args.workers,
                        epochs=args.epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        default="data/driving_log.csv",
                        type=str,
                        help="path to csv (default: data/driving_log.csv)")

    parser.add_argument("--model",
                        default="model.h5",
                        type=str,
                        help="model (default: model.h5)")

    parser.add_argument("--epoch",
                        default=10,
                        type=int,
                        help="Number of epochs (default: 10)")

    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Minibatch size (default: 32)")

    parser.add_argument("--workers",
                        default=8,
                        type=int,
                        help="Number of Threads for data generator (default: 4)")

    args = parser.parse_args()

    main(args)
