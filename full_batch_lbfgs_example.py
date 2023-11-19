"""
Full-Batch L-BFGS Implementation with Wolfe Line Search

Demonstrates how to implement a simple full-batch L-BFGS with weak Wolfe line search 
without Powell damping to train a simple convolutional neural network using the LBFGS 
optimizer.

This implementation is CUDA-compatible.

Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
Last edited 10/20/20.

Requirements:
    - Keras (for CIFAR-10 dataset)
    - NumPy
    - PyTorch

Run Command:
    python full_batch_lbfgs_example.py

"""
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18 as _resnet18
from einops import rearrange, reduce, repeat

from tensorflow.keras.datasets import mnist, cifar10 # to load dataset

from functions.utils import compute_stats, get_grad
from functions.LBFGS import FullBatchLBFGS

# Parameters for L-BFGS training
max_iter = 200                      # note each iteration is NOT an epoch
ghost_batch = 512
batch_size = 512
overlap_ratio = 0.25                # should be in (0, 0.5)
lr = 1

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255
if len(X_train.shape) == 3:
    X_train = repeat(X_train, 'b h w -> b c h w', c = 3)
    X_test = repeat(X_test, 'b h w -> b c h w', c = 3)
else:
    X_train = rearrange(X_train, 'b h w c -> b c h w')
    X_test = rearrange(X_test, 'b h w c -> b c h w')
# X_train = np.transpose(X_train, (0, 3, 1, 2))
# X_test = np.transpose(X_test, (0, 3, 1, 2))

# Define network
def resnet18(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet18(pretrained=pretrained, **kwargs)
    return model

# Check cuda availability
cuda = torch.cuda.is_available()

# Create neural network model
if cuda:
    torch.cuda.manual_seed(2018)
    model = resnet18().cuda()
else:
    torch.manual_seed(2018)
    model = resnet18()

# Define helper functions

# Forward pass
if cuda:
    opfun = lambda X: model.forward(torch.from_numpy(X).cuda())
else:
    opfun = lambda X: model.forward(torch.from_numpy(X))

# Forward pass through the network given the input
if cuda:
    predsfun = lambda op: np.argmax(op.cpu().data.numpy(), 1)
else:
    predsfun = lambda op: np.argmax(op.data.numpy(), 1)

# Do the forward pass, then compute the accuracy
accfun = lambda op, y: np.mean(np.equal(predsfun(op), y.squeeze())) * 100

# Define optimizer
optimizer = FullBatchLBFGS(model.parameters(), lr=1., history_size=10, line_search='Wolfe', debug=True)

# Main training loop
no_samples = X_train.shape[0]

# compute initial gradient and objective
grad, obj = get_grad(optimizer, X_train, y_train, opfun)

# main loop
for n_iter in range(max_iter):

    # training mode
    model.train()

    # define closure for line search
    def closure():

        optimizer.zero_grad()

        if cuda:
            loss_fn = torch.tensor(0, dtype=torch.float).cuda()
        else:
            loss_fn = torch.tensor(0, dtype=torch.float)

        for subsmpl in np.array_split(np.arange(no_samples), max(int(no_samples / ghost_batch), 1)):

            ops = opfun(X_train[subsmpl])

            if cuda:
                tgts = torch.from_numpy(y_train[subsmpl]).cuda().long().squeeze()
            else:
                tgts = torch.from_numpy(y_train[subsmpl]).long().squeeze()

            loss_fn += F.cross_entropy(ops, tgts) * (len(subsmpl) / no_samples)

        return loss_fn

    # perform line search step
    options = {'closure': closure, 'current_loss': obj}
    obj, grad, lr, _, _, _, _, _ = optimizer.step(options)

    # compute statistics
    model.eval()
    train_loss, test_loss, test_acc = compute_stats(X_train, y_train, X_test, y_test, opfun, accfun,
                                                    ghost_batch=128)

    # print data
    print('Iter:', n_iter + 1, 'lr:', lr, 'Training Loss:', train_loss, 'Test Loss:', test_loss,
          'Test Accuracy:', test_acc)
