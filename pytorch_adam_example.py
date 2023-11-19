import torch
import torchvision
import torchvision.transforms as transforms

import math
import time

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# try replacing relu with elu
torch.manual_seed(42)
default_batch=128 # no. of batches per epoch 50000/default_batch
batches_for_report=10#

transform=transforms.Compose(
   [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='./torchdata', train=True,
    download=True, transform=transform)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=default_batch,
    shuffle=True, num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./torchdata', train=False,
    download=True, transform=transform)

testloader=torch.utils.data.DataLoader(testset, batch_size=default_batch,
    shuffle=False, num_workers=0)

classes=('plane', 'car', 'bird', 'cat', 
  'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


#####################################################
def verification_error_check(net):
   correct=0
   total=0
   for data in testloader:
     images,labels=data
     outputs=net(Variable(images).to(mydevice))
     _,predicted=torch.max(outputs.data,1)
     correct += (predicted==labels.to(mydevice)).sum()
     total += labels.size(0)

   return 100*correct//total
#####################################################

net = torchvision.models.resnet18(pretrained=False).cuda()

# loss function and optimizer
import torch.optim as optim
# from lbfgsnew import LBFGSNew # custom optimizer
criterion=nn.CrossEntropyLoss()
#optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer=optim.Adam(net.parameters(), lr=0.001)
# optimizer = LBFGSNew(net.parameters(), history_size=7, max_iter=2, line_search_fn=True,batch_mode=True)

start_time=time.time()
use_lbfgs=True
# train network
for epoch in range(20):
  running_loss=0.0
  for i,data in enumerate(trainloader,0):
    # get the inputs
    inputs,labels=data
    # wrap them in variable
    inputs,labels=Variable(inputs).to(mydevice),Variable(labels).to(mydevice)

    if not use_lbfgs:
     # zero gradients
     optimizer.zero_grad()
     # forward+backward optimize
     outputs=net(inputs)
     loss=criterion(outputs,labels)
     loss.backward()
     optimizer.step()
    else:
      def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        if loss.requires_grad:
          loss.backward()
          #print('loss %f l1 %f l2 %f'%(loss,l1_penalty,l2_penalty))
        return loss
      optimizer.step(closure)
    # only for diagnostics
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    running_loss +=loss.data.item()

    if math.isnan(loss.data.item()):
       print('loss became nan at %d'%i)
       break

    # print statistics
    if i%(batches_for_report) == (batches_for_report-1): # after every 'batches_for_report'
      print('%f: [%d, %5d] loss: %.5f accuracy: %.3f'%
         (time.time()-start_time,epoch+1,i+1,running_loss/batches_for_report,
         verification_error_check(net)))
      running_loss=0.0

print('Finished Training')


# save model (and other extra items)
# torch.save({
#             'model_state_dict':net.state_dict(),
#             'epoch':epoch,
#             'optimizer_state_dict':optimizer.state_dict(),
#             'running_loss':running_loss,
#            },'./res.model')


# whole dataset
correct=0
total=0
for data in trainloader:
   images,labels=data
   outputs=net(Variable(images).to(mydevice)).cpu()
   _,predicted=torch.max(outputs.data,1)
   total += labels.size(0)
   correct += (predicted==labels).sum()
   
print('Accuracy of the network on the %d train images: %d %%'%
    (total,100*correct//total))

correct=0
total=0
for data in testloader:
   images,labels=data
   outputs=net(Variable(images).to(mydevice)).cpu()
   _,predicted=torch.max(outputs.data,1)
   total += labels.size(0)
   correct += (predicted==labels).sum()
   
print('Accuracy of the network on the %d test images: %d %%'%
    (total,100*correct//total))


class_correct=list(0. for i in range(10))
class_total=list(0. for i in range(10))
for data in testloader:
  images,labels=data
  outputs=net(Variable(images).to(mydevice)).cpu()
  _,predicted=torch.max(outputs.data,1)
  c=(predicted==labels).squeeze()
  for i in range(4):
    label=labels[i]
    class_correct[label] += c[i]
    class_total[label] += 1

for i in range(10):
  print('Accuracy of %5s : %2d %%' %
    (classes[i],100*float(class_correct[i])/float(class_total[i])))