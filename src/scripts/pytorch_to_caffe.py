import os

import numpy as np
import torch
import torchvision
from torchvision.models import resnet
from pytorch2caffe import pytorch2caffe
import sys
sys.path.append('/home/star/Desktop/goturn-pytorch/src/')
from goturn.network.network import GoturnNetwork
import matplotlib.pyplot as plt
from multiprocessing import Manager


def SaveDemo():
    from torchvision.models import resnet

    name = 'resnet18'
    resnet18 = GoturnNetwork()
    checkpoint = torch.load("/home/star/Desktop/goturn-pytorch/src/scripts/caffenet/epoch=0-step=32.pth")
    resnet18.load_state_dict(checkpoint['state_dict'],strict=False)
    # resnet18 = resnet.resnet18()
    resnet18.eval()

    # dummy_input = torch.ones([1, 3, 224, 224])
    dummy_input1 =  torch.load("prev.pt")
    dummy_input2 = torch.load("curr.pt")

    # pytorch2caffe.trans_net(resnet18, dummy_input1, dummy_input2,name)
    # pytorch2caffe.trans_net(resnet18, dummy_input1,dummy_input2, name)
    # pytorch2caffe.save_prototxt('{}.prototxt'.format(name))
    # pytorch2caffe.save_caffemodel('{}.caffemodel'.format(name))


class IA:
    def __set_name__(self, owner, name):
        print("set name invoke")
        self.pf = name

    def __set__(self, instance, value):
        print("set invoke")
        if isinstance(value,str):
            print("get the attr of string",value)
        else:
            # setattr(instance,self.pf,value)
            print("other attr got")

    def __get__(self, instance, owner):
        pass
        # value = getattr(instance, self.pf)
        # return value
class A:
    descriptor = IA()

    def __init__(self):
        self.descriptor = self.func

    def func(self):
        print("store the function")

def test():
    return [4,5,6]

class GoturnDataloader:
    def __init__(self,images_p = None):
        self._images_p = images_p

    def collate(self,batch):
        print(self._images_p)

if __name__ == '__main__':
    # SaveDemo()
    # a = A()

    manager = Manager()
    objGoturn = GoturnDataloader(images_p=manager.list())

    list1 = [1,2]
    list2 = [3,4]
    list3 = [5,6]
    list4 = [7,8]
    list5 = [list1,list2,list3,list4]
    for batch in list5:
        objGoturn.collate(batch)



