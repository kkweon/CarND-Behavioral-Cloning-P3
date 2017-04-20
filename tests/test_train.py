import model
import numpy as np


file_path = "./data/driving_log.csv"


def test_load_data():
    X, y = model.load_data(file_path)

    assert type(X) == np.ndarray
    assert X[0] == "data/IMG/center_2016_12_01_13_30_48_287.jpg"
    N = X.shape[0]
    assert y.shape == (N, )


def test_read_csv():

    data = model.read_csv(file_path)

    assert data[0][0] == "IMG/center_2016_12_01_13_30_48_287.jpg"
    assert data[0][1] == "IMG/left_2016_12_01_13_30_48_287.jpg"
    assert data[0][-1] == 22.14829
    assert data[2][-1] == 1.453011


def test_generator():
    X, y = model.load_data(file_path)
    X, y = next(model.generator(X, y, train=False))

    assert type(X) == np.ndarray
    assert X.shape[1:] == (160, 320, 3)
    N = len(X)
    assert y.shape == (N, )
