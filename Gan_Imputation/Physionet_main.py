#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:53 2018

@author: yonghong, luo
"""
from __future__ import print_function
import sys
sys.path.append("mnt/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks/GRUI")
import WGAN_GRUI 
import tensorflow as tf
import argparse #used to parse command line arguments and convert them to usable variables
import numpy as np
from Physionet2012Data import readData, readTestData
import os

"""main"""
def main():
    #parse arguments
    #This creates command line arguments (variables) that will be passed to the WGAN. 
    #This is useful because we can change any parameter from  the command line when we run the code without having to change the code itself.
    #This creates ease of use when trying to increase the performance of the WGAN by finding the right parameter values.
    #Example: python Physionet_main.py --batch-size 64
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--impute-iter', type=int, default=400)
    parser.add_argument('--pretrain-epoch', type=int, default=5)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path', type=str, default="../set-a/")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--g-loss-lambda',type=float,default=0.1)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)  #When the learning rate (lr) is 0.001, the pretrain_loss decreases quickly, and 4 epochs are sufficient.
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--n-inputs', type=int, default=41)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)  #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)
    parser.add_argument('--isSlicing',type=int,default=1)
    parser.add_argument('--disc-iters',type=int,default=8)
    args = parser.parse_args()

    #This converts these integer command line arguments to boolean values.
    #This makes the code more readable and easier to use.
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True
    if args.isNormal==0:
            args.isNormal=False
    if args.isNormal==1:
            args.isNormal=True
    if args.isSlicing==0:
            args.isSlicing=False
    if args.isSlicing==1:
            args.isSlicing=True

    #Make the max step length of two datasets the same.
    #Initialize hyperparameters for WGAN training.
    epochs=[30]  #Indicates the number of training epochs.
    g_loss_lambdas=[0.15]  #Indicates the weight of the generator loss, affects the rate at which the generator learns.  Increasing value causes faster learning compared to the discriminator.
    beta1s=[0.5]  #Indicates the momentum parameter for the Adam optimizer, helps stabalize the training of the GAN and prevent getting stuck in a local minima.

    #Loop over all possible combinations of the previously defined hyperparameters.
    for beta1 in beta1s:
        for e in epochs:
            for g_l in g_loss_lambdas:
                args.epoch=e
                args.beta1=beta1
                args.g_loss_lambda=g_l
                tf.reset_default_graph()
                dt_train=readData.ReadPhysionetData(os.path.join(args.data_path,"train"), os.path.join(args.data_path,"train","list.txt"),isNormal=args.isNormal,isSlicing=args.isSlicing)
                dt_test=readTestData.ReadPhysionetData(os.path.join(args.data_path,"test"), os.path.join(args.data_path,"test","list.txt"),dt_train.maxLength,isNormal=args.isNormal,isSlicing=args.isSlicing)
                tf.reset_default_graph()
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True 
                with tf.Session(config=config) as sess:
                    gan = WGAN_GRUI.WGAN(sess,
                                args=args,
                                datasets=dt_train,
                                )
            
                    # build graph
                    gan.build_model()
            
                    # show network architecture
                    #show_all_variables()
            
                    # launch the graph in a session
                    gan.train()
                    print(" [*] Training finished!")
                    
                    gan.imputation(dt_train,True)
                    
                    print(" [*] Train dataset Imputation finished!")
                    
                    gan.imputation(dt_test,False)
                    
                    print(" [*] Test dataset Imputation finished!")
                tf.reset_default_graph()
if __name__ == '__main__':
    main()
