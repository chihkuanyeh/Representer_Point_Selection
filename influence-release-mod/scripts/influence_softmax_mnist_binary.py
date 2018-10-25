#!/usr/bin/env python
# coding: utf-8
import time
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import numpy as np
import math
import pickle
import os
import torch
import torch.nn as nn
dtype = torch.cuda.DoubleTensor


class softmax(nn.Module):
    """Support Vector Machine"""
    def __init__(self, W):
        super(softmax, self).__init__()
        self.W = Variable(torch.from_numpy(W).type(dtype), requires_grad=True)
        #print(self.W)

    def forward(self, x, y):
        D = (torch.matmul(x,self.W))
        max_value,_ = torch.max(D,1,keepdim = True)
        A = torch.log(torch.sum(torch.exp(D-max_value),dim = 1))
        B = torch.sum(D*y,dim=1)
        Phi = torch.sum(A-B)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return (Phi,L2)

def to_np(x):
    return x.data.cpu().numpy()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.9 every 10 epochs"""
    lr = args.lr * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def backtracking_line_search(optimizer,model,grad,x,y,val,beta,N,args):
    t = 1.0
    W_O = to_np(model.W)
    grad_np = to_np(grad)
    while(True):
        model.W = Variable(torch.from_numpy(W_O-t*grad_np).type(dtype), requires_grad=True)
        val_n = 0.0
        (Phi,L2) = model(x,y)
        val_n = Phi/N + L2*args.lmbd
        if t < 0.000001 :
            break
        if to_np(val_n - val + t*torch.norm(grad)**2/2)[0]>=0:
            t = beta *t
        else:
            break

def train(X, Y, model, args, Y_true = None):
    X = torch.DoubleTensor(X)
    Y = torch.DoubleTensor(Y)
    x = Variable(X.cuda())
    y = Variable(Y.cuda())
    N = len(Y)
    min_loss = 10000.0
    optimizer = optim.SGD([model.W],lr = 1.0)
    for epoch in range(args.epoch):
        sum_loss = 0
        phi_loss = 0
        optimizer.zero_grad()
        (Phi,L2) = model(x,y)
        loss = L2*args.lmbd + Phi/N
        phi_loss += to_np(Phi/N)[0]
        loss.backward()
        z = model.W.grad
        temp_W = model.W.data
        backtracking_line_search(optimizer,model,model.W.grad,x,y,loss,0.6,N,args)
        grad_loss = to_np(torch.mean(torch.abs(z)))[0]
        if grad_loss < min_loss:
            if epoch ==0:
                init_grad = grad_loss
            min_loss = grad_loss
            best_W = temp_W
            #print('grad')
            #print(min_loss)
            #print('phi_loss')
            #print((phi_loss))
            if min_loss < init_grad/2000000:
                print('reached')
                print(epoch)
                sys.stdout.flush()
                break
        if epoch % 100 == 0:
            print('Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}'.format(epoch, to_np(loss)[0], phi_loss, grad_loss))
            #sys.stdout.flush()
    temp = torch.matmul(x,Variable(best_W.cuda()))
    max_value,_ = torch.max(temp,1,keepdim = True)
    D_exp = torch.exp(temp-max_value)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N,1)
    #print(D_exp_sum.shape)
    #print(D_exp.shape)
    weight_matrix = D_exp.div(D_exp_sum.expand_as(D_exp))-y
    weight_matrix = torch.div(weight_matrix,(-2.0*args.lmbd*N))
    w = torch.matmul(torch.t(x),weight_matrix)
    X = Variable(X.cuda())
    Y = Variable(Y.cuda())
    temp = torch.matmul(x,w.cuda())
    max_value,_ = torch.max(temp,1,keepdim = True)
    D_exp = torch.exp(temp-max_value)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N,1)
    y_p = D_exp.div(D_exp_sum.expand_as(D_exp))
    temp = torch.matmul(x,Variable(best_W.cuda()))
    max_value,_ = torch.max(temp,1,keepdim = True)
    D_exp = torch.exp(temp-max_value)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N,1)
    y_pp = D_exp.div(D_exp_sum.expand_as(D_exp))
    print('diff')
    y_p = to_np(y_p)
    y_pp = to_np(y_pp)
    y_p[np.where(y_p<0.5)]=0
    y_p[np.where(y_p>0.5)]=1
    y_p = 1-y_p
    y_pp[np.where(y_pp>0.5)]=1
    y_pp[np.where(y_pp<0.5)]=0
    print(to_np(Y)[:20,:])
    print(y_p[:20,:])
    print(y_pp[:20,:])
    print(np.mean(np.abs(to_np(Y)-y_pp)))
    print(np.mean(np.abs(y_p-y_pp)))
    #print(np.mean(np.abs(Y_true-y_pp)))
    #print(np.mean(np.abs(Y_true-y_p)))
    if np.mean(np.abs(y_p-y_pp))>0.08:
        input()
    sys.stdout.flush()
    return to_np(weight_matrix)

def main(args):
    data = np.load('mnist_binary_1vs7_small.npz')
    x_train = data['x_train']
    y_train1 = data['y_train']
    y_true = np.zeros((y_train1.shape[0],2))
    print('ground truth')
    print(y_train1[:20])
    y_true[np.where(y_train1==1),1] = 1
    y_true[np.where(y_train1==-1),0] = 1
    print(y_true[:20,:])

    for i in range(5):
        data = np.load('mnist_1vs7_corrupt_{}.npz'.format(i))
        x_train = data['x_train']
        x_test = data['x_test']
        y_train1 = data['y_train']
        y_train = np.zeros((y_train1.shape[0],2))
        print(y_train1[:20])
        y_train[np.where(y_train1==1),1] = 1
        y_train[np.where(y_train1==-1),0] = 1
        print(y_train[:20,:])
        weight = np.load('logistic_weight_{}.npy'.format(i))
        weight = np.zeros(weight.shape)
        print('done loading')
        model = softmax(weight.T)
        model.cuda()
        X = np.concatenate([x_train,np.ones((x_train.shape[0],1))],axis = 1)
        Y = y_train
        print(X.shape)
        print(Y.shape)
        print(weight.shape)
        print(np.matmul(X,weight.T)[:20,:])
        print(Y[:20,:])
        print(y_true[:20,:])
        weight_matrix = train(X,Y, model, args, y_true)
        with open("weight_matrix_mnist_binary_test_{}.pkl".format(i), "wb") as output_file:
            pickle.dump([weight_matrix,y_train], output_file, protocol=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbd', type=float, default=0.001)
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=30000)
    args = parser.parse_args()
    print(args)
    main(args)
