import torch
import platform
from six.moves import cPickle as pickle
import numpy as np
import os

@torch.no_grad()
def sample(image, label, batch_shape, stop=None, ftype=torch.float):
    image = image.to(ftype)
    label = label.to(torch.long)
    count = 0
    while True:
        if stop and count == stop: break
        index = torch.randint(size=batch_shape, low=0, high=image.shape[0])
        batch_image = image[index]
        batch_noise = batch_image[torch.randint(0, batch_image.shape[0], size=(batch_image.shape[0], ))] * ((torch.rand_like(batch_image) > 0.5)*1 - (torch.rand_like(batch_image) > 0.5)*1) * 0.1
        batch_label= label[index]
        yield batch_image + batch_noise, batch_label
        count += 1

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

# load CIFAR tensor
def load_cifar_tensor(trn_size, tst_size, device):
    x_train, y_train, x_test, y_test = load_CIFAR10('./input/cifar-10-batches-py')
    if trn_size == None: trn_size = x_train.shape[0]
    if tst_size == None: tst_size = x_test.shape[0]
    return tuple(map(
        lambda x: torch.from_numpy(x).to(device), 
        (x_train[:trn_size], y_train[:trn_size], x_test[:tst_size], y_test[:tst_size])
    ))


if __name__ == '__main__':
    trn_image, trn_label, tst_image, tst_label = load_cifar_tensor(None, -1, 'cpu')
    for x, y in sample(trn_image, trn_label, batch_shape=(13,11)):
        print(x.shape)
        print(y.shape)