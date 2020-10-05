# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 02:46:22 2019

@author: User
"""
import numpy as np
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import copy
import multiprocessing
import pickle
import os

def imbalanced_data(dataset,data_loader, cls_num,cls_prob,n_cpu,batch_size):

    cls_num=list(map(int,cls_num.split('_')))
    cls_prob=list(map(float,cls_prob.split('_')))
    
    assert len(cls_num) == len(cls_prob)
    idx_candi=np.array([]).astype(int)

    len_list=[]
    bin_dict={}
    for num_candi in range(len(cls_num)):
        if opt.dataset == 'MNIST':
            candi=np.where(data_loader.dataset.targets.numpy() == cls_num[num_candi])[0]
            num_of_choice=int(np.around(len(candi) * cls_prob[num_candi]))
            len_list.append(num_of_choice)
            idx_=np.random.choice(candi, num_of_choice, replace=False)
            bin_dict[cls_num[num_candi]] = idx_
            idx_candi=np.append(idx_candi,idx_)
            
        elif opt.dataset == 'cifar10':
            candi=np.where(torch.as_tensor(data_loader.dataset.targets).numpy() == int(cls_num[num_candi]))[0]
            num_of_choice=int(np.around(len(candi) * cls_prob[num_candi]))
            len_list.append(num_of_choice)
            idx_=np.random.choice(candi, num_of_choice, replace=False)
            bin_dict[cls_num[num_candi]] = idx_
            idx_candi=np.append(idx_candi,idx_)

    new_dataset=copy.deepcopy(dataset)
    
    if opt.dataset == 'MNIST':
        new_dataset.targets = torch.from_numpy(data_loader.dataset.targets.numpy()[idx_candi])
        new_dataset.data = torch.from_numpy(data_loader.dataset.data.numpy()[idx_candi])

    elif opt.dataset == 'cifar10':
        new_dataset.targets = torch.as_tensor(data_loader.dataset.targets).numpy()[idx_candi]
        new_dataset.data = torch.as_tensor(data_loader.dataset.data).numpy()[idx_candi]
            
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    print('class :', cls_num)
    print('num_of_sample :',len_list)
    print('num_of_sample :',np.around(np.array(len_list) / np.sum(np.array(len(data_loader.dataset))),4))
    print('total_data :', len(data_loader.dataset))
    return new_dataset,new_data_loader,bin_dict


if __name__ == '__main__':
        
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='cifar10 | mnist', default='cifar10')
    parser.add_argument('--dataroot', help='path to dataset', default='./download_data')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cls_num', help='list_of_numbers')
    parser.add_argument('--imb_ratio', type=str,help='imb_ratio')
    
    opt = parser.parse_args()
    print(opt)
    
    n_cpu= multiprocessing.cpu_count()
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opt.cls_num)
    
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    
    elif opt.dataset == 'MNIST':
        train_dataset = datasets.MNIST(root=opt.dataroot, download=True, train=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]))


    assert train_dataset
    
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=n_cpu,drop_last=True)

    imbalanced_dataset,imbalanced_data_loader,imbalanced_dict=imbalanced_data(train_dataset, dataloader, opt.cls_num, opt.imb_ratio, n_cpu, opt.batchSize)

    dict_path=r'./dict/'
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)
        
    with open(dict_path + opt.cls_num + '_'+opt.dataset, 'wb') as filename:
        pickle.dump(imbalanced_dict, filename)
