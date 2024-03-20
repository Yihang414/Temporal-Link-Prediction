import os
import math
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from loader import *
from util import *
from model import Generator, Discriminator



def load_args():
    parser = argparse.ArgumentParser()
    # seed & device
    parser.add_argument('--device_no', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
    #dataset
    parser.add_argument('--link_path', type=str, default='./data/Amount.npy', help='directory of link dataset')
    parser.add_argument('--node_features_path', type=str, default='./data/population_2000-2022.npy', help='directory of node dataset')
    parser.add_argument('--window_size', type=int, default=5, help="history time window size")
    parser.add_argument('--node_num', type=int, default=128, help="number of the node in graph")
    parser.add_argument('--max_thres', type=int, default=1e6, help="max thres of the graph")
    parser.add_argument('--ratio', type=int, default=1e6, help="the ratio to shrink the multiple of the value")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="the ratio to split the dataset")
    # model
    parser.add_argument('--in_feature', type=int, default=1)
    parser.add_argument('--out_feature', type=int, default=256)
    parser.add_argument('--lstm_feature', type=int, default=32)
    parser.add_argument('--disc_hidden', type=int, default=128)
    parser.add_argument('--epsilon', type=float, default=0.1)
    # train
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--pretrain_epoches', type=int, default=100)
    parser.add_argument('--gan_epoches', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')


#optimizer
    parser.add_argument('--pretrain_learning_rate', type=float, default=0.1)
    parser.add_argument('--d_learning_rate', type=float, default=0.001)
    parser.add_argument('--g_learning_rate', type=float, default=0.001)
    parser.add_argument('--gradient_clip', type=float, default=5.0)
    parser.add_argument('--weight_clip', type=float, default=0.001)
    
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")
    
    return args

def pretrain(args, generator, train_loader):
    generator.train()
    for epoch in range(args.pretrain_epoches):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            in_shots, out_shot = data
            in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
            predicted_shot = generator(in_shots)
            predicted_shot = predicted_shot.squeeze(0).reshape(-1, args.node_num, args.node_num)
            out_shot = out_shot.reshape(-1, args.node_num, args.node_num)#.view(args.batch_size, -1)
            args.pretrain_optimizer.zero_grad()
            loss = args.criterion(predicted_shot, out_shot)
            nn.utils.clip_grad_norm_(generator.parameters(), args.gradient_clip)
            loss.backward()
            args.pretrain_optimizer.step()
            epoch_loss += loss.item()
        print('[epoch %d] [loss %.4f]' % (epoch+1, epoch_loss))

def train(args, generator, discriminator, train_loader, sample_loader):
    generator.train() 
    discriminator.train()
    for i, (data, sample) in enumerate(zip(train_loader, sample_loader)):
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
        _, sample = sample
        sample = sample.cuda()
        sample = sample.view(args.batch_size, -1)
        predicted_shot = generator(in_shots)
        predicted_shot = predicted_shot.squeeze(0)
        real_logit = discriminator(sample).mean()
        fake_logit = discriminator(predicted_shot).mean()
        
        # update discriminator
        args.discriminator_optimizer.zero_grad()
        discriminator_loss = - real_logit + fake_logit
        discriminator_loss.backward(retain_graph=True)
        args.discriminator_optimizer.step()
        for p in discriminator.parameters():
            p.data.clamp_(-args.weight_clip, args.weight_clip)
        
        # update generator
        args.generator_optimizer.zero_grad()
        generator_loss = -fake_logit
        generator_loss.backward()
        args.generator_optimizer.step()
        out_shot = out_shot.view(args.batch_size, -1)
        mse_loss = args.criterion(predicted_shot, out_shot)
        print('[epoch %d] [step %d] [d_loss %.4f] [g_loss %.4f] [mse_loss %.4f]' % (args.epoch+1, i,
                discriminator_loss.item(), generator_loss.item(), mse_loss.item()))

def eval(args, generator, test_loader):
    total_samples = 0
    total_mse = 0
    total_kl = 0
    total_missrate = 0
    
    for i, data in enumerate(test_loader):
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
        predicted_shot = generator(in_shots)
        predicted_shot = predicted_shot.squeeze(0).reshape(-1, args.node_num, args.node_num)
        predicted_shot = (predicted_shot + predicted_shot.transpose(1, 2)) / 2
        for j in range(args.node_num):
            predicted_shot[:, j, j] = 0
        mask = predicted_shot >= args.epsilon
        predicted_shot = predicted_shot * mask.float()
        batch_size = in_shots.size(0)
        total_samples += batch_size
        total_mse += batch_size * MAE(predicted_shot, out_shot)
        total_kl += batch_size * EdgeWiseKL(predicted_shot, out_shot)
        total_missrate += batch_size * MissRate(predicted_shot, out_shot)
    
    print('MSE: %.4f' % (total_mse / total_samples))
    print('edge wise KL: %.4f' % (total_kl / total_samples))
    print('miss rate: %.4f' % (total_missrate / total_samples))

def main(args):
    set_seed(args.seed)
    
    adj_reg, node_features = load_data_with_features(args.link_path, args.node_features_path, args.ratio, args.max_thres)
    dataset = TradeLinkDataset(adj_reg, node_features, args.window_size)
    
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=args.train_ratio)
    
    if args.train:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers = args.num_workers)
        sample_loader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=args.num_workers)
        args.criterion = MAE
        
        ## define a model and load chek points
        generator = Generator(
            window_size=args.window_size,
            node_num=args.node_num,
            in_features=args.in_feature,
            out_features=args.out_feature,
            lstm_features=args.lstm_feature
        )

        discriminator = Discriminator(input_size=args.node_num * args.node_num, hidden_size=args.disc_hidden)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        args.pretrain_optimizer = optim.RMSprop(generator.parameters(), lr=args.pretrain_learning_rate)
        args.generator_optimizer = optim.RMSprop(generator.parameters(), lr=args.g_learning_rate)
        args.discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=args.d_learning_rate)
        print('Start pretrain !')
        pretrain(args, generator, train_loader)
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        eval(args, generator, val_loader)
        
    #     print('Start GAN training !')
    #     for epoch in range(args.gan_epoches):
    #         args.epoch = epoch
    #         train(args, generator, discriminator, train_loader, sample_loader)
    #     print('Train finished !')
    #     torch.save(generator, os.path.join('./save_model', 'generator.pkl'))
    # generator = torch.load(os.path.join('./save_model', 'generator.pkl')).cuda()
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # eval(args, generator, val_loader)


if __name__ == "__main__":
    args = load_args()
    main(args)