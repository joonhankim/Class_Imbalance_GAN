# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 16:32:03 2019

@author: seonghee
"""
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pylab as plt
import cnn_model
import time
import multiprocessing
import argparse
import pandas as pd
import pickle
import copy
import dataset_test
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score


def indexing_data(dataset,data_loader, idx_array,n_cpu):
    
    new_dataset=copy.deepcopy(dataset)
    if opt.dataset == 'MNIST':
        new_dataset.targets = torch.from_numpy(data_loader.dataset.targets.numpy()[idx_array])
        new_dataset.data = torch.from_numpy(data_loader.dataset.data.numpy()[idx_array])

    elif opt.dataset == 'cifar10':
        new_dataset.targets = torch.as_tensor(data_loader.dataset.targets).numpy()[idx_array]
        new_dataset.data = torch.as_tensor(data_loader.dataset.data).numpy()[idx_array]
            
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    if opt.dataset == 'MNIST':
        print('class :', np.unique(new_dataset.targets.numpy()))

    elif opt.dataset == 'cifar10':
        print('class :', np.unique(new_dataset.targets))
    
#    print('dataset :', dataset)
#    print('num_of_sample :',len(new_data_loader.dataset))

    return new_dataset, new_data_loader


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

def train(model, trn_loader, device, criterion, optimizer):
    model.train()

    running_loss = 0
    running_acc = 0
    for i,data in enumerate(trn_loader):
        
        input,label=data
        input,label=input.to(device), label.to(device, dtype=torch.int64)
        model.zero_grad()
        
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        #print statistics
        _, pred= torch.max(output.data, 1)
        accuracy = (pred == label).sum().item() / label.size(0)
        
        running_loss += loss.item()
        running_acc += accuracy
        
    trn_loss = running_loss/(i+1)
    trn_acc = running_acc/(i+1)
    return trn_loss, trn_acc


def validation(model, vali_loader, device, criterion):
    model.eval()

    running_loss = 0
    running_acc = 0

    for i, data in enumerate(vali_loader):
        input,label=data
        #print(label)
        input,label=input.to(device), label.to(device, dtype=torch.int64)
        output = model(input)
        loss = criterion(output, label)

        _, pred= torch.max(output.data, 1)
        accuracy = (pred == label).sum().item() / label.size(0)

        running_loss += loss.item()
        running_acc += accuracy

    vali_loss = running_loss/(i+1)
    vali_acc =  running_acc/(i+1)
    return vali_loss, vali_acc

def test(model, tst_loader, device, criterion):
    model.eval()
    
    running_loss = 0
    running_acc = 0
    running_f1 = 0
    running_g_mean = 0
    for i, data in enumerate(tst_loader):
        input,label = data
        input, label = input.to(device), label.to(device,dtype=torch.int64)

        output = model(input)
        loss = criterion(output, label)

        _, pred= torch.max(output.data, 1)
        accuracy = (pred == label).sum().item() / label.size(0)

        real_bi_numpy=label.cpu().numpy()
        real_bi_numpy[real_bi_numpy == pos_label] = 1
        real_bi_numpy[real_bi_numpy == neg_label] = 0
        
        pred_bi_numpy=pred.cpu().numpy()
        pred_bi_numpy[pred_bi_numpy == pos_label] = 1
        pred_bi_numpy[pred_bi_numpy == neg_label] = 0
        
        f1=np.around(f1_score(real_bi_numpy, pred_bi_numpy, pos_label=1,average='binary'),4)
        g_mean=np.around(geometric_mean_score(real_bi_numpy, pred_bi_numpy, pos_label=1,average='binary'),4)

        print(np.unique(pred.cpu().numpy(), return_counts=True))
        

        running_loss += loss.item()
        running_acc += accuracy
        running_f1 += f1
        running_g_mean += g_mean

    tst_loss = running_loss / (i+1)
    tst_acc=running_acc / (i+1)
    tst_f1 = running_f1/(i+1)
    tst_g_mean = running_g_mean/(i+1)
    return tst_loss, tst_acc,tst_f1, tst_g_mean


def data_split(dataset,validation_split):
    data=dataset
    split_per = validation_split
    dataset_size = len(data)
    
    indices = list(range(dataset_size))
    
    split = int(np.floor(split_per * dataset_size))
    np.random.seed(1337)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    return train_sampler,valid_sampler
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',help='cifar10 | MNIST')
    parser.add_argument('--dataroot', help='path to dataset', default='./download_data')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--cls_num', help='list_of_numbers')
    parser.add_argument('--cls_prob', type=str)
    parser.add_argument('--clf_model', type=str)
    parser.add_argument('--GAN_model', type=str, help='AC_GAN, Normal_GAN,NO_GAN, DC_GAN')
    parser.add_argument('--test_batch', type=int, default=128)
    opt = parser.parse_args()
    print(opt)
    
    pos_label=int(opt.cls_num.split('_')[0])
    neg_label=int(opt.cls_num.split('_')[1])
    
    n_cpu= multiprocessing.cpu_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opt.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./download_data',train=True, download=True,transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./download_data', train=False,download=True,transform=transform)


    elif opt.dataset == 'MNIST':
        transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, ),(0.5, ))])    
        trainset = torchvision.datasets.MNIST(root='./download_data',train=True, download=True,transform=transform)
        testset = torchvision.datasets.MNIST(root='./download_data', train=False,download=True,transform=transform)


    testloader=torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,shuffle=True, num_workers=n_cpu,drop_last=True)
    
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,shuffle=True, num_workers=n_cpu,drop_last=True)


    ###
    dict_path=r'./dict/'
    new_dict_path = dict_path + opt.cls_num + '_' + opt.dataset
    with open(new_dict_path, 'rb') as filename:
        dictionary=pickle.load(filename)
    ###
    whole_index=dictionary[list(dictionary.keys())[0]].tolist()+dictionary[list(dictionary.keys())[1]].tolist()
    ###
    imbalanced_dataset,imbalanced_data_loader=indexing_data(trainset,dataloader,whole_index,n_cpu)
    ###
    if opt.GAN_model != 'NO_GAN':

        image_folder_path='./oversampled_data/'+opt.GAN_model+'/'+opt.dataset+'/'+opt.cls_num+'/'
        if opt.dataset == 'MNIST':
            Train_dataset_A = dataset_test.MNIST(data_dir=image_folder_path)
        
        elif opt.dataset == 'cifar10':
            Train_dataset_A = dataset_test.cifar10(data_dir=image_folder_path)
        
        concat_dataset = ConcatDataset((Train_dataset_A, imbalanced_dataset))

        final_train_dataset = concat_dataset

    ###
    elif opt.GAN_model == 'NO_GAN':
        final_train_dataset = imbalanced_dataset


    ###임의로 설정
    train_sampler, valid_sampler = data_split(final_train_dataset, 0.3)

    train_loader = torch.utils.data.DataLoader(final_train_dataset,batch_size=opt.batchSize,sampler=train_sampler,num_workers=n_cpu)
    valid_loader = torch.utils.data.DataLoader(final_train_dataset,batch_size=opt.batchSize, sampler=valid_sampler, num_workers=n_cpu)
    
    imbalanced_test,imbalanced_test_loader,test_dict=imbalanced_data(testset,testloader,opt.cls_num,opt.cls_prob,n_cpu, opt.test_batch)


    if opt.clf_model == 'Lenet' :
                
        if opt.dataset == 'cifar10':
            model_net = torch.nn.DataParallel(cnn_model.LeNet(), device_ids=[0])
        elif opt.dataset == 'MNIST':
            model_net = torch.nn.DataParallel(cnn_model.LeNet_M(), device_ids=[0])

    elif opt.clf_model == 'Resnet':
        if opt.dataset == 'cifar10':
            model_net=torch.nn.DataParallel(cnn_model.ResNet(cnn_model.BasicBlock, [2,2,2,2]),device_ids=[0])

        elif opt.dataset == 'MNIST':
            model_net = torch.nn.DataParallel(cnn_model.ResNet_M(cnn_model.BasicBlock, [2,2,2,2]), device_ids=[0]) 

        
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model_net.parameters(), lr=0.01, momentum=0.9)

    dir_path = './clf_models/'+opt.dataset+opt.cls_num+'/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    #training
    trn_acc_list=[]
    trn_loss_list=[]
    vali_acc_list=[]
    vali_loss_list=[]
    
    c_time=0
    
    min_loss=float('inf')
    for epoch in range(opt.niter):
        start=time.time()
        trn_loss, trn_acc = train(model_net, train_loader, device, criterion, optimizer)
        vali_loss, vali_acc = validation(model_net, valid_loader, device, criterion)

        trn_loss_list.append(trn_loss)
        trn_acc_list.append(trn_acc)
        vali_loss_list.append(vali_loss)
        vali_acc_list.append(vali_acc)

        c_time += time.time() - start
        
        print('epoch : %s trn_loss : % .4f trn_acc : % .4f vali_loss : %.4f vali_acc : %.4f time : %.4f c_time : %.4f'%(epoch+1, trn_loss, trn_acc, vali_loss, vali_acc, time.time() - start, c_time))
        
        if vali_loss < min_loss:
            min_loss = vali_loss
            torch.save(model_net.state_dict(), dir_path+opt.GAN_model+'_'+opt.clf_model+'.pkl')
            
    loss_df=pd.DataFrame({'trn_acc' : trn_acc_list, 'trn_loss' : trn_loss_list, 'vali_acc' : vali_acc_list, 'vali_loss' : vali_loss_list})
    loss_df.to_csv(dir_path+opt.GAN_model+'_'+opt.clf_model+'.csv', index=False)

    #train vali acc
    plt.plot(np.arange(len(trn_acc_list)), trn_acc_list,label='trn_acc',c='r',linewidth=4)
    plt.plot(np.arange(len(vali_acc_list)), vali_acc_list,label='vali_acc',c='g',linewidth=4,linestyle='dashed')
    plt.title(opt.dataset+'_'+opt.clf_model+'_'+opt.cls_num+'_'+'acc')
    plt.legend()
    plt.show()
    plt.savefig(dir_path+opt.GAN_model+'_'+opt.clf_model+'_'+'acc.png')
    plt.close()
    
    
    #train vali loss
    plt.plot(np.arange(len(trn_loss_list)), trn_loss_list, label='trn_loss',c='r',linewidth=4)
    plt.plot(np.arange(len(vali_loss_list)), vali_loss_list,label='vali_loss',c='g',linewidth=4,linestyle='dashed')
    plt.title(opt.dataset+'_'+opt.clf_model+'_'+opt.cls_num+'_'+'loss')
    plt.legend()
    plt.show()
    plt.savefig(dir_path+opt.GAN_model+'_'+opt.clf_model+'_'+'loss.png')
    plt.close()
    
    
    if opt.clf_model == 'Lenet' :
                
        if opt.dataset == 'cifar10':
            new_model_net = torch.nn.DataParallel(cnn_model.LeNet(), device_ids=[0])
        elif opt.dataset == 'MNIST':
            new_model_net = torch.nn.DataParallel(cnn_model.LeNet_M(), device_ids=[0])

    elif opt.clf_model == 'Resnet':
        if opt.dataset == 'cifar10':
            new_model_net=torch.nn.DataParallel(cnn_model.ResNet(cnn_model.BasicBlock, [2,2,2,2]),device_ids=[0])

        elif opt.dataset == 'MNIST':
            new_model_net = torch.nn.DataParallel(cnn_model.ResNet_M(cnn_model.BasicBlock, [2,2,2,2]), device_ids=[0]) 

    new_model_net.load_state_dict(torch.load(dir_path+opt.GAN_model+'_'+opt.clf_model+'.pkl'))
    new_model_net.eval()

    
    #testset accuracy, loss 확인
    test_loss, test_acc, test_f1, test_g_mean = test(new_model_net, imbalanced_test_loader, device, criterion)
    print('test_loss : %.4f test_acc : % .4f test_f1 : % .4f test_g_mean : % .4f'%(test_loss, test_acc, test_f1, test_g_mean))
    final_result=pd.DataFrame({'test_loss' : [test_loss], 'test_acc' : [test_acc], 'test_f1' : [test_f1], 'test_g_mean' : test_g_mean})
    final_result.to_csv(dir_path+opt.GAN_model+'_'+opt.clf_model+'_'+'final_result.csv', index=False)
    
