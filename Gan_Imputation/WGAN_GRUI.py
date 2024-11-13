# This is the Wasserstein Generative Adversarial Network that will generate the imputed values for the missing data.
# It uses a Gated Recurrent Unit (GRU) to assist in processing the time series data.  Usage of the GRU allows for the capturing of the temporal depencies within the data.
# The main difference between this and a traditional GAN is the replacement of the loss function from binary cross entropy to the Wasserstein distance.

#-*- coding: utf-8 -*-
from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from ops import *
from utils import *
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("mnt/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks/GRUI")
#sys.path.append("..")
from GRUI import mygru_cell

"""
Discriminator input normalization, do not use m, fill with 0
Generator input: remove m, only keep delta.
g does not accumulate z every time.
"""
class WGAN(object):
    model_name = "WGAN_no_mask"     # name for checkpoint

    #WGAN constructor: establishes arguments needed when the class is called and creates required variables for the remaining methods of the class.
    #All args. calls are taking the default value passed in the Physionet_main.py file or whatever modification is passed via the command line when it is executed. 
    def __init__(self, sess, args, datasets):
        self.sess = sess 
        self.isbatch_normal=args.isBatch_normal       # Determine wheter to apply batch normalization.
        self.isNormal=args.isNormal                   # Determine wheter to apply batch normalization.
        self.checkpoint_dir = args.checkpoint_dir     # Directory to save checkpoint.
        self.result_dir = args.result_dir             # Directory to save results.
        self.log_dir = args.log_dir                   # Directory to save log.
        self.dataset_name=args.dataset_name           # Dataset name
        self.run_type=args.run_type                   # Type of run (training/test)
        self.lr = args.lr                             # Learning Rate
        self.epoch = args.epoch                       # Number of epoch (training evolutions, how many times the entire dataset is processed).
        self.batch_size = args.batch_size             # Training batch size (the number of data points that will be trained).
        self.n_inputs = args.n_inputs                 # This is the number of inputs/features the network will accept (data input size). MNIST data input (img shape: 28*28).
        self.n_steps = datasets.maxLength             # This represents the time steps used in the time series.
        self.n_hidden_units = args.n_hidden_units     # This is the number of neurons in hidden layer.
        self.n_classes = args.n_classes               # This is the number of output classes.  MNIST classes (0-9 digits).
        self.gpus=args.gpus                           # This identifies which GPUs are available for use during training.
        self.run_type=args.run_type                   # Type of run (training/test).
        self.result_path=args.result_path             # Specifies the directory where the results will be saved.
        self.model_path=args.model_path               # This is the path to the location where trained models or checkpoints will be stored, allowing for model saving and loading.
        self.pretrain_epoch=args.pretrain_epoch       # Defines how many epochs the model will be pre-trained.
        self.impute_iter=args.impute_iter             # Defines how many times the imputation process is repeated.
        self.isSlicing=args.isSlicing                 # Determines whether the data should be sliced into smaller sections for processing (used for performance management).
        self.g_loss_lambda=args.g_loss_lambda         # Regularization parameter for the generator loss.
        
        self.datasets=datasets                        # This stores the dataset object for accessing the training data.
        self.z_dim = args.z_dim                       # Dimension of noise-vector (noise vector = what the generator uses to create the fake data)
        self.gen_length=args.gen_length               # Designates how many samples the generator should produce each sequence (i.e. weekly time series should have gen_length == 7)
        
        # WGAN_GP parameter
        self.lambd = 0.25                             # WGAN gradient penalty.  The higher value, the more stable, but the slower convergence.
        self.disc_iters = args.disc_iters             # The number of discriminator iterations for one-step of generator.

        # train
        # This includes a TensorFlow version check and implements different mygru_cells from mygru_cell.py depending on the TF version.
        # The TF version check isn't really useful anymore as all of these versions are outdated.  We are using the TF version 1.7 managed through the Docker Desktop Application.
        # The mygru_cell is used for processing the time series data.
        self.learning_rate = args.lr
        self.beta1 = args.beta1
        if "1.5" in tf.__version__ or "1.7" in tf.__version__ :
            self.grud_cell_d_fw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_d_bw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g_fw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g_bw = mygru_cell.MyGRUCell15(self.n_hidden_units)
        elif "1.4" in tf.__version__:
            self.grud_cell_d_fw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_d_bw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g_fw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g_bw = mygru_cell.MyGRUCell15(self.n_hidden_units)
        elif "1.2" in tf.__version__:
            self.grud_cell_d_fw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_d_bw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g_fw = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g_bw = mygru_cell.MyGRUCell15(self.n_hidden_units)
        # test
        self.sample_num = 64  # This establishes the number of generated samples (images) to be saved at each evaluation phase.  Used to evaluate the generators performance visually.

        self.num_batches = len(datasets.x) // self.batch_size  #Establishes how many iterations the model will run through the dataset in each epoch.  The total number of batches available for training.

    # This method pretrains the generator to generate plausible data based on the training data.
    # X = input, M = mask matrix(identifies which values are missing and which are present), Delta = time gap
    def pretrainG(self, X, M, Delta,  Mean, Lastvalues, X_lengths, Keep_prob, is_training=True, reuse=False):
        
        with tf.variable_scope("g_enerator", reuse=reuse):                                     # Defines scope of the the variables. Ensures all variables created within scope are tied to the generators model.  
            
            """
            the rnn cell's variable scope is defined by tensorflow,
            if we want to update rnn cell's weights, the variable scope must contains 'g_' or 'd_'            
            """
            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())       # Defines the weight matrix for the input data.
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.random_normal_initializer())   # Defines the weight matrix for the output predictions.

            # These lines initialize the biases for both the hidden and output layers.
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))                  # Initialize the bias for the input data.
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))                      # Initialize the bias for the output predicitons.
            
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.random_normal_initializer())                  # Initialize the weight matrix for the generator.
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))                           # Initialize the bias for the generator.
            
            X = tf.reshape(X, [-1, self.n_inputs])                                                                                    # Input reshaped to match the dimensionality for further operations.
            Delta=tf.reshape(Delta,[-1,self.n_inputs])                                                                                # Time gap variable reshape to match the dimensionality 
                                                                                                                                      # for further operations.

            rth= tf.matmul(Delta, wr_h)+br_h                                                # Calculate the decay factor based on time gap and input weights.
            rth=math_ops.exp(-tf.maximum(0.0,rth))                                          # Applies transformation to model the effecto fo time gaps between consecuitive observations.
            
            X=tf.concat([X,rth],1)                                                          # Decay factor is concatenated with the original input.
            
            X_in = tf.reshape(X, [-1, self.n_steps, self.n_inputs+self.n_hidden_units])     # The previous combined matrix is reshaped to fit the format for the RNN.
         
            init_state_fw = self.grud_cell_g_fw.zero_state(self.batch_size, dtype=tf.float32)     # initialize hidden cell of Gated Reccurent Unit GRU with an all-zero state
            init_state_bw = self.grud_cell_g_bw.zero_state(self.batch_size, dtype=tf.float32)   
          
            # Runs the RNN using the Gated Recurrent Unit (GRU) over the input and return outputs for each time step as well as the final hidden state.
            # outpus stores the output of the RNN at each step, final_state stores the last hidden state.
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(self.grud_cell_g_fw, 
                                self.grud_cell_g_bw,                             
                                X_in, \
                                initial_state_fw=init_state_fw,\
                                initial_state_bw=init_state_bw,                                  
                                sequence_length=X_lengths,
                                time_major=False)
          
            #outputs: batch_size*n_steps*n_hiddensize
            #The output is reshaped and the final predictions are produced.
            outputs_fw, outputs_bw = outputs 
            outputs = tf.add(outputs_fw, outputs_bw) / 2.0
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])                            # The RNN output is reshaped and the final predictions are produced and then reshaped to match the original input.
            
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out          # Generates predictions based on the reshaped RNN outputs. Dropout function helps prevent overfitting.
            out_predict=tf.reshape(out_predict,[-1,self.n_steps,self.n_inputs])             # Reshapes output predictions to 3D tensor.
            return out_predict                                                              # Returns the predictions for the missing values.

    #Implements the discriminator for the WGAN    
    def discriminator(self, X, M, DeltaPre, Lastvalues ,DeltaSub ,SubValues , Mean,  X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("d_iscriminator", reuse=reuse):  # Defines scope of the the variables. Ensures all variables created within scope are tied to the generators model. 

            # wr_h and w_out define the weights for the input data and the final output predictions.  w_out is the weight matrix for the output predictions.
            # br_h and b_out initialize the biases for both the hidden and output layers. b_out is the bias matrix for the output predictions.
            wr_h=tf.get_variable("d_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("d_w_out",shape=[self.n_hidden_units, 1],initializer=tf.random_normal_initializer())
            br_h= tf.get_variable("d_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("d_b_out",shape=[1, ],initializer=tf.constant_initializer(0.001))
          
           
            M=tf.reshape(M,[-1,self.n_inputs])  #Mask matrix reshaped to match the dimensionality for further operations.
            X = tf.reshape(X, [-1, self.n_inputs])  #Input reshaped to match the dimensionality for further operations.
            DeltaPre=tf.reshape(DeltaPre,[-1,self.n_inputs])  #Time gap variable reshaped to match the dimensionality for further operations.
           
            # These lines produce the decay factor which models the effect of time gaps between consecutive observations in the time series.
            rth= tf.matmul(DeltaPre, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            
            # add noise
            #X=X+np.random.standard_normal(size=(self.batch_size*self.n_steps, self.n_inputs))/100 
            X=tf.concat([X,rth],1)  # Decay factor is concatenated with the original input.
              
            X_in = tf.reshape(X, [self.batch_size, self.n_steps , self.n_inputs+self.n_hidden_units])  #The previous combined matrix is reshaped to fit the format for the RNN.
            
            init_state_fw = self.grud_cell_d_fw.zero_state(self.batch_size, dtype=tf.float32) # initialize with an all-zero state
            init_state_bw = self.grud_cell_d_bw.zero_state(self.batch_size, dtype=tf.float32)
          
            #runs the Gated Recurrent Unit over the input and return outputs for each time step as well as the final hidden state.
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(self.grud_cell_d_fw, 
                                                     self.grud_cell_d_bw,
                                                     X_in, \
                                                     initial_state_fw=init_state_fw,\
                                                     initial_state_bw=init_state_bw,
                                                     sequence_length=X_lengths,
                                                     time_major=False)

            # Average forward and backward directions for final state
            final_state_fw, final_state_bw = final_states
            final_state = tf.add(final_state_fw, final_state_bw) / 2.0
         
            # final_state:batch_size*n_hiddensize
            # Cannot use the last one, should use the length one. Previously, the last one was used, so the output was always b_out regardless."
            out_logit=tf.matmul(tf.nn.dropout(final_state,Keep_prob), w_out) + b_out  #Produces a single value that indicates whether the input data is real or fake.
            out =tf.nn.sigmoid(out_logit)    # Select the last output.  Produces a probability score between 0 and 1.  Values closer to 1 indicate real data.
            return out,out_logit  #Returns the probability for being real (out) and the indicator for whether the input data is real or fake.
            
    # Implements the generator
    def generator(self, z, Keep_prob, is_training=True, reuse=False):
        # x,delta,n_steps
        # z :[self.batch_size, self.z_dim]
        # first feed noise in rnn, then feed the previous output into next input
        # or we can feed noise and previous output into next input in future version
        with tf.variable_scope("g_enerator", reuse=reuse):  # Defines scope of the the variables. Ensures all variables created within scope are tied to the generators model.
            #generate 

            # wr_h and w_out define the weights for the input data and the final output predictions.  w_out is the weight matrix for the output predictions.
            # br_h and b_out initialize the biases for both the hidden and output layers. b_out is the bias matrix for the output predictions.
            # w_z and b_z are weights for mapping the noise vector z to the input space.
            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.random_normal_initializer())
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))            
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.random_normal_initializer())
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            
            #self.times=tf.reshape(self.times,[self.batch_size,self.n_steps,self.n_inputs])
            #change z's dimension
            # batch_size*z_dim-->batch_size*n_inputs
            x=tf.matmul(z,w_z)+b_z  # These lines map the noise vector z and creates the startin input for the generator.
            x=tf.reshape(x,[-1,self.n_inputs]) 
            delta_zero=tf.constant(0.0,shape=[self.batch_size,self.n_inputs]) 
            #delta_normal=tf.constant(48.0*60.0/self.gen_length,shape=[self.batch_size,self.n_inputs])
            #delta:[batch_size,1,n_inputs]
            

            # combine X_in
            # These lines produce the decay factor which models the effect of time gaps between consecutive observations in the time series.
            rth= tf.matmul(delta_zero, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            
            x=tf.concat([x,rth],1)            
            X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])
            
            init_state_fw = self.grud_cell_g_fw.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            init_state_bw = self.grud_cell_g_bw.zero_state(self.batch_size, dtype=tf.float32)
          
            #z=tf.reshape(z,[self.batch_size,1,self.z_dim])
            seq_len=tf.constant(1,shape=[self.batch_size])
            
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(self.grud_cell_g_fw, 
                                                      self.grud_cell_g_bw,
                                                      X_in, \
                                                      initial_state_fw=init_state_fw,\
                                                      initial_state_bw=init_state_bw,
                                                      sequence_length=seq_len,
                                                      time_major=False)

            final_state_fw, final_state_bw = final_states
            init_state=tf.add(final_state_fw, final_state_bw) / 2.0
          
            #outputs: batch_size*1*n_hidden
            outputs_fw, outputs_bw = outputs
            outputs = tf.add(outputs_fw, outputs_bw) / 2.0
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
          
            # full connect
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
            
            total_result=tf.multiply(out_predict,1.0)
            
            for i in range(1,self.n_steps):
                out_predict=tf.reshape(out_predict,[self.batch_size,self.n_inputs])
                # optionally add noise z to the output 
                #out_predict=out_predict+tf.matmul(z,w_z)+b_z
                #
                delta_normal=tf.reshape(self.imputed_deltapre[:,i:(i+1),:],[self.batch_size,self.n_inputs])
                rth= tf.matmul(delta_normal, wr_h)+br_h
                rth=math_ops.exp(-tf.maximum(0.0,rth))

                # concatenate the current output and the decay term, prepare it for the GRU input
                x=tf.concat([out_predict,rth],1)
                X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])

                # run the GRU for the current step
                outputs, final_states = tf.nn.bidirectional_dynamic_rnn(self.grud_cell_g_fw, 
                                                          self.grud_cell_g_bw,
                                                          X_in, \
                                                          initial_state_fw=init_state_fw,\
                                                          initial_state_bw=init_state_bw,
                                                          sequence_length=seq_len,
                                                          time_major=False)
              
                # Average forward and backward directions for final state
                final_state_fw, final_state_bw = final_states
                init_state=tf.add(final_state_fw, final_state_bw) / 2.0 # update state for next step

                outputs_fw, outputs_bw = outputs
                outputs = tf.add(outputs_fw, outputs_bw) / 2.0
                outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
                out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
                out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
                total_result=tf.concat([total_result,out_predict],1) # concatenate outputs across steps
            
            #delta:[batch_size,,n_inputs]
            # if batch normalization is enabled, apply it to the result
            if self.isbatch_normal:
                with tf.variable_scope("g_bn", reuse=tf.AUTO_REUSE):
                    total_result=bn(total_result,is_training=is_training, scope="g_bn_imple")
            
            # return the final generated sequence and various intermediate values for some imputation
            last_values=tf.multiply(total_result,1)
            sub_values=tf.multiply(total_result,1)

            return total_result,self.imputed_deltapre,self.imputed_deltasub,self.imputed_m,self.x_lengths,last_values,sub_values
        
    def impute(self):
        with tf.variable_scope("impute", reuse=tf.AUTO_REUSE):
            z_need_tune=tf.get_variable("z_needtune",shape=[self.batch_size,self.z_dim],initializer=tf.random_normal_initializer(mean=0,stddev=0.1) )
            return z_need_tune
            
    def build_model(self):
        # these are all just placeholders for various inputs and parameters
        self.keep_prob = tf.placeholder(tf.float32) 
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.mean = tf.placeholder(tf.float32, [self.n_inputs,])
        self.deltaPre = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.deltaSub = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.subvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.x_lengths = tf.placeholder(tf.int32,  shape=[self.batch_size,])
        self.imputed_deltapre=tf.placeholder(tf.float32,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        self.imputed_deltasub=tf.placeholder(tf.float32,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        self.imputed_m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        
        

        """ Loss Function """
        # pretrain the generator G to produce outputs close to the real data
        Pre_out=self.pretrainG(self.x, self.m, self.deltaPre,  self.mean,\
                                                      self.lastvalues, self.x_lengths,self.keep_prob, \
                                                      is_training=True, reuse=False)
        
        # pretraining loss is based on the difference between generated vs real data
        self.pretrain_loss=tf.reduce_sum(tf.square(tf.multiply(Pre_out,self.m)-self.x)) / tf.cast(tf.reduce_sum(self.x_lengths),tf.float32)
        
        #discriminator( X, M, DeltaPre, Lastvalues ,DeltaSub ,SubValues , Mean,  X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):
        
        # real discriminator D_real output is based on real input data
        D_real, D_real_logits = self.discriminator(self.x, self.m, self.deltaPre,self.lastvalues,\
                                                   self.deltaSub,self.subvalues,  self.mean,\
                                                       self.x_lengths,self.keep_prob, \
                                                      is_training=True, reuse=False,isTdata=True)

        #G return total_result,self.imputed_deltapre,self.imputed_deltasub,self.imputed_m,self.x_lengths,last_values,sub_values
        # fake discriminator D_fake which is based on fake, generated input data
        
        g_x,g_deltapre,g_deltasub,g_m,G_x_lengths,g_last_values,g_sub_values = self.generator(self.z,self.keep_prob, is_training=True, reuse=True)
        
        D_fake, D_fake_logits = self.discriminator(g_x,g_m,g_deltapre,g_last_values,\
                                                   g_deltasub,g_sub_values,self.mean,\
                                                      G_x_lengths,self.keep_prob,
                                                      is_training=True, reuse=True ,isTdata=False)
        
        """
        impute loss
        """
        # fine tuning the noise variable for imputation
        self.z_need_tune=self.impute()

        # generated output during imputation 
        impute_out,impute_deltapre,impute_deltasub,impute_m,impute_x_lengths,impute_last_values,impute_sub_values=self.generator(self.z_need_tune,self.keep_prob, is_training=False, reuse=True)
        
        # discriminator output for the imputed data
        impute_fake, impute_fake_logits = self.discriminator(impute_out,impute_m,impute_deltapre,impute_last_values,\
                                                             impute_deltasub,impute_sub_values,self.mean,\
                                                      impute_x_lengths,self.keep_prob,
                                                      is_training=False, reuse=True ,isTdata=False)
        
        # loss for imputation, combines reconstruction and adversarial loss
      
        # Wasserstein (Original)
        # self.impute_loss=tf.reduce_mean(tf.square(tf.multiply(impute_out,self.m)-self.x))-self.g_loss_lambda*tf.reduce_mean(impute_fake_logits)

        # Pearson Chi Squared (Modification)
        # self.impute_loss = tf.reduce_mean(tf.square(tf.multiply(impute_out, self.m) - self.x)) - self.g_loss_lambda * tf.reduce_mean(0.25 * impute_fake_logits**2 + impute_fake_logits)

        # Forward KL (Modification)
        self.impute_loss = tf.reduce_mean(tf.square(tf.multiply(impute_out, self.m) - self.x)) - self.g_loss_lambda * tf.reduce_mean(tf.exp(impute_fake_logits - 1))

        # Reverse KL (Modification)
        # self.impute_loss = tf.reduce_mean(tf.square(tf.multiply(impute_out, self.m) - self.x)) - self.g_loss_lambda * tf.reduce_mean(-1 - impute_fake_logits)

        # Squared Hellinger
        # self.impute_loss = tf.reduce_mean(tf.square(tf.multiply(impute_out, self.m) - self.x)) - self.g_loss_lambda * tf.reduce_mean((1 - tf.exp(impute_fake_logits)) / (tf.exp(impute_fake_logits)))
      
        # final imputed data is a mixture between real and impputed data, decided by mask m
        self.impute_out=impute_out
        
        #the imputed results
        self.imputed=tf.multiply((1-self.m),self.impute_out)+self.x
      
        # get loss for discriminator

        # Wasserstein (Original)
        # d_loss_real = - tf.reduce_mean(D_real_logits)
        # d_loss_fake = tf.reduce_mean(D_fake_logits)
        # self.d_loss = d_loss_real + d_loss_fake
      
        # Pearson Chi Squared (Modification)
        # d_loss_real = - tf.reduce_mean(D_real_logits)
        # d_loss_fake = tf.reduce_mean(0.25*D_fake_logits**2 + D_fake_logits)
        # self.d_loss = d_loss_real + d_loss_fake

        # Forward KL (Modification)
        d_loss_real = -tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(tf.exp(D_fake_logits - 1))
        self.d_loss = d_loss_real + d_loss_fake

        # Reverse KL (Modification)
        # d_loss_real = -tf.reduce_mean(-tf.exp(D_real_logits))
        # d_loss_fake = tf.reduce_mean(-1 - D_fake_logits)
        # self.d_loss = d_loss_real + d_loss_fake

        # Squared Hellinger
        # d_loss_real = -tf.reduce_mean(1 - tf.exp(D_real_logits))
        # d_loss_fake = tf.reduce_mean((1 - tf.exp(D_fake_logits)) / (tf.exp(D_fake_logits)))
        # self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator

        # Wasserstein Distance (Original)
        # self.g_loss = - d_loss_fake

        # Pearson Chi Squared (Modification)
        # self.g_loss = -tf.reduce_mean(0.25*D_fake_logits**2 + D_fake_logits)

        # Forward KL (Modification)
        self.g_loss = -tf.reduce_mean(tf.exp(D_fake_logits - 1))

        # Reverse KL (Modification)
        # self.g_loss = -tf.reduce_mean(-1 - D_fake_logits)

        # Squared Hellinger
        # self.g_loss = -tf.reduce_mean((1 - tf.exp(D_fake_logits)) / (tf.exp(D_fake_logits)))
        
        

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        z_vars = [self.z_need_tune]
        '''
        print("d vars:")
        for v in d_vars:
            print(v.name)
        print("g vars:")
        for v in g_vars:
            print(v.name)
        print("z vars:")
        for v in z_vars:
            print(v.name)
        '''
        
        #don't need normalization because we have adopted the dropout
        """
        ld = 0.0
        for w in d_vars:
            ld += tf.contrib.layers.l2_regularizer(1e-4)(w)
        lg = 0.0
        for w in g_vars:
            lg += tf.contrib.layers.l2_regularizer(1e-4)(w)
        
        self.d_loss+=ld
        self.g_loss+=lg
        """
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # this code have used batch normalization, so the upside line should be executed
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.d_loss, var_list=d_vars)
            #self.d_optim=self.optim(self.learning_rate, self.beta1,self.d_loss,d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*self.disc_iters, beta1=self.beta1) \
                        .minimize(self.g_loss, var_list=g_vars)
            #self.g_optim=self.optim(self.learning_rate, self.beta1,self.g_loss,g_vars)
            self.g_pre_optim=tf.train.AdamOptimizer(self.learning_rate*2,beta1=self.beta1) \
                        .minimize(self.pretrain_loss,var_list=g_vars)
        self.impute_optim=tf.train.AdamOptimizer(self.learning_rate*7,beta1=self.beta1) \
                    .minimize(self.impute_loss,var_list=z_vars)
    
        
        

        #clip weight
        self.clip_all_vals = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in t_vars]
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in d_vars]
        self.clip_G = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in g_vars]
        
        
        """" Testing """
        # for test
        self.fake_x,self.fake_delta,_,_,_,_,_ = self.generator(self.z, self.keep_prob, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        g_pretrain_loss_sum=tf.summary.scalar("g_pretrain_loss", self.pretrain_loss)
        # final summary operations
        self.impute_sum=tf.summary.scalar("impute_loss", self.impute_loss)
        self.g_sum = g_loss_sum
        self.g_pretrain_sum=tf.summary.merge([g_pretrain_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum,d_loss_fake_sum, d_loss_sum])

    #learning_rate = controls step size for updating weights     
    def optim(self,learning_rate,beta,loss,var):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta)
        grads = optimizer.compute_gradients(loss,var_list=var)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
        train_op = optimizer.apply_gradients(grads)
        return train_op
    def pretrain(self, start_epoch,counter,start_time):
        
        if start_epoch < self.pretrain_epoch:
            #todo
            for epoch in range(start_epoch, self.pretrain_epoch):
            # get batch data
                self.datasets.shuffle(self.batch_size,True)
                idx=0
                #x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
                for data_x,data_y,data_mean,data_m,data_deltaPre,data_x_lengths,data_lastvalues,_,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in self.datasets.nextBatch():
                    
                    # pretrain
                    _, summary_str, p_loss = self.sess.run([self.g_pre_optim, self.g_pretrain_sum, self.pretrain_loss],
                                                   feed_dict={self.x: data_x,
                                                              self.m: data_m,
                                                              self.deltaPre: data_deltaPre,
                                                              self.mean: data_mean,
                                                              self.x_lengths: data_x_lengths,
                                                              self.lastvalues: data_lastvalues,
                                                              self.deltaSub:deltaSub,
                                                              self.subvalues:subvalues,
                                                              self.imputed_m:imputed_m,
                                                              self.imputed_deltapre:imputed_deltapre,
                                                              self.imputed_deltasub:imputed_deltasub,
                                                              self.keep_prob: 0.5})
                    self.writer.add_summary(summary_str, counter)
    
    
                    counter += 1
    
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, pretrain_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, p_loss))
                    idx+=1
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                #调好之后再保存
                #if epoch%10==0:
                #    self.save(self.checkpoint_dir, counter)


    def train(self):
      # Generate random noise samples for training
      self.sample_z = np.random.standard_normal(size=(self.batch_size, self.z_dim))

      # Create a TensorFlow saver to save the model's checkpoints
      self.saver = tf.train.Saver()

      # Initialize a summary writer for TensorBoard logging
      self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name + '/' + self.model_dir)

      # Attempt to load the latest checkpoint, if it exists
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          # Calculate the starting epoch based on the checkpoint counter
          start_epoch = (int)(checkpoint_counter / self.num_batches)
          # Start batch ID is set to zero for simplicity
          start_batch_id = 0
          # Counter is initialized based on the start epoch
          counter = start_epoch * self.num_batches
          print(" [*] Load SUCCESS")
          return 
      else:
          # If no checkpoint is found, initialize all TensorFlow variables
          tf.global_variables_initializer().run()
          start_epoch = 0  # Starting from the first epoch
          start_batch_id = 0  # Starting from the first batch
          counter = 1  # Initial counter
          print(" [!] Load failed...")

      # Start measuring time for the training process
      start_time = time.time()
    
      # Call pretraining function, if applicable
      self.pretrain(start_epoch, counter, start_time)
      if start_epoch < self.pretrain_epoch:
          start_epoch = self.pretrain_epoch  # Ensure pretraining is completed if needed
    
      # Main training loop over epochs
      for epoch in range(start_epoch, self.epoch):
          # Shuffle the dataset for each epoch
          self.datasets.shuffle(self.batch_size, True)
          idx = 0  # Initialize index for batches
        
          # Loop over batches of data
          for data_x, data_y, data_mean, data_m, data_deltaPre, data_x_lengths, data_lastvalues, _, imputed_deltapre, imputed_m, deltaSub, subvalues, imputed_deltasub in self.datasets.nextBatch():
            
              # Generate random noise for the generator input
              batch_z = np.random.standard_normal(size=(self.batch_size, self.z_dim))
              # Clip the values of the discriminator to enforce gradient penalties
              _ = self.sess.run(self.clip_all_vals)
              # Train the discriminator and get the loss and summary
              _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                             feed_dict={self.z: batch_z,
                                                        self.x: data_x,
                                                        self.m: data_m,
                                                        self.deltaPre: data_deltaPre,
                                                        self.mean: data_mean,
                                                        self.x_lengths: data_x_lengths,
                                                        self.lastvalues: data_lastvalues,
                                                        self.deltaSub: deltaSub,
                                                        self.subvalues: subvalues,
                                                        self.imputed_m: imputed_m,
                                                        self.imputed_deltapre: imputed_deltapre,
                                                        self.imputed_deltasub: imputed_deltasub,
                                                        self.keep_prob: 0.5})
              # Log the discriminator summary to TensorBoard
              self.writer.add_summary(summary_str, counter)

              # Update the generator network every `disc_iters` iterations
              if counter % self.disc_iters == 0:
                  # Train the generator and get the loss and summary
                  _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], 
                                                         feed_dict={self.z: batch_z,
                                                         self.keep_prob: 0.5,
                                                         self.deltaPre: data_deltaPre,
                                                         self.mean: data_mean,
                                                         self.x_lengths: data_x_lengths,
                                                         self.lastvalues: data_lastvalues,
                                                         self.deltaSub: deltaSub,
                                                         self.subvalues: subvalues,
                                                         self.imputed_m: imputed_m,
                                                         self.imputed_deltapre: imputed_deltapre,
                                                         self.imputed_deltasub: imputed_deltasub,
                                                         self.mean: data_mean})
                  # Log the generator summary to TensorBoard
                  self.writer.add_summary(summary_str, counter)
                  # Print training status with current epoch, batch info, and losses
                  print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, counter: %4d" \
                    % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, counter))

              # Increment the counter for training steps
              counter += 1

              # Display training status for every batch processed
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, counter: %4d" \
                    % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, counter))

              # Save generated samples every 300 steps
              if np.mod(counter, 300) == 0:
                  # Generate fake samples from the generator
                  fake_x, fake_delta = self.sess.run([self.fake_x, self.fake_delta],
                                           feed_dict={self.z: batch_z,
                                                      self.deltaPre: data_deltaPre,
                                                      self.mean: data_mean,
                                                      self.x_lengths: data_x_lengths,
                                                      self.lastvalues: data_lastvalues,
                                                      self.deltaSub: deltaSub,
                                                      self.subvalues: subvalues,
                                                      self.imputed_m: imputed_m,
                                                      self.imputed_deltapre: imputed_deltapre,
                                                      self.imputed_deltasub: imputed_deltasub,
                                                      self.mean: data_mean,
                                                      self.keep_prob: 0.5})
                  # If running in training mode, save the generated samples
                  if self.run_type == "train":
                      self.writeG_Samples("G_sample_x", counter, fake_x)
                      self.writeG_Samples("G_sample_delta", counter, fake_delta)
                
              idx += 1  # Increment the batch index
            
          # Reset the start_batch_id to zero after each epoch
          start_batch_id = 0

      # Save the model checkpoint after training is complete
      self.save(self.checkpoint_dir, counter)

    def imputation(self, dataset, isTrain):
        # Set the dataset for imputation and shuffle it
        self.datasets = dataset
        self.datasets.shuffle(self.batch_size, True)
        # Initialize the variable needed for tuning
        tf.variables_initializer([self.z_need_tune]).run()
        # The training data can't be perfectly divisible by batch_size; the rest is discarded
        start_time = time.time()  # Start timing the imputation process
        batchid = 1  # Initialize batch ID
        impute_tune_time = 1  # Initialize the impute tuning time
        counter = 1  # Initialize counter for imputation steps
    
        # Loop over batches for imputation
        for data_x, data_y, data_mean, data_m, data_deltaPre, data_x_lengths, data_lastvalues, _, imputed_deltapre, imputed_m, deltaSub, subvalues, imputed_deltasub in self.datasets.nextBatch():
            # Initialize the z variable for tuning
            tf.variables_initializer([self.z_need_tune]).run()
        
            # Run imputation for a set number of iterations
            for i in range(0, self.impute_iter):
                # Perform optimization step for imputation
                _, impute_out, summary_str, impute_loss, imputed = self.sess.run([self.impute_optim, self.impute_out, self.impute_sum, self.impute_loss, self.imputed],
                                                       feed_dict={self.x: data_x,
                                                       self.m: data_m,
                                                       self.deltaPre: data_deltaPre,
                                                       self.mean: data_mean,
                                                       self.x_lengths: data_x_lengths,
                                                       self.lastvalues: data_lastvalues,
                                                       self.deltaSub: deltaSub,
                                                       self.subvalues: subvalues,
                                                       self.imputed_m: imputed_m,
                                                       self.imputed_deltapre: imputed_deltapre,
                                                       self.imputed_deltasub: imputed_deltasub,
                                                       self.keep_prob: 1.0})
                impute_tune_time += 1  # Increment the impute tuning time
                counter += 1  # Increment the counter
            
                # Print status every 10 iterations
                if counter % 10 == 0:
                    print("Batchid: [%2d] [%4d/%4d] time: %4.4f, impute_loss: %.8f" \
                          % (batchid, impute_tune_time, self.impute_iter, time.time() - start_time, impute_loss))
                    # Log summary to TensorBoard
                    self.writer.add_summary(summary_str, counter / 10)
        
            # Save the imputed results
            self.save_imputation(imputed, batchid, data_x_lengths, data_deltaPre, data_y, isTrain)
            batchid += 1  # Increment batch ID
            impute_tune_time = 1  # Reset impute tuning time for the next batch

    @property
    def model_dir(self):
        # Generate a string representing the model directory based on various parameters
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.epoch, self.disc_iters,
            self.batch_size, self.z_dim,
            self.lr, self.impute_iter,
            self.isNormal, self.isbatch_normal,
            self.isSlicing, self.g_loss_lambda,
            self.beta1
            )

    def save_imputation(self, impute_out, batchid, data_x_lengths, data_times, data_y, isTrain):
        # Determine the directory for saving imputation results based on training or testing
        if isTrain:
            imputation_dir = "imputation_train_results"
        else:
            imputation_dir = "imputation_test_results"
    
        # Create the necessary directories if they don't exist
        if not os.path.exists(os.path.join(imputation_dir,
                                     self.model_name,
                                     self.model_dir)):
            os.makedirs(os.path.join(imputation_dir, 
                                     self.model_name,
                                     self.model_dir))
        
        # Write the imputed data
        resultFile = open(os.path.join(imputation_dir,
                                       self.model_name,
                                       self.model_dir,
                                       "batch" + str(batchid) + "x"), 'w')
        for length in data_x_lengths:
            resultFile.writelines(str(length) + ",")  # Write lengths of each series
        resultFile.writelines("\r\n")
    
        # Write imputed output data
        for oneSeries in impute_out:
            resultFile.writelines("begin\r\n")  # Mark the beginning of a series
            for oneClass in oneSeries:
                for i in oneClass.flat:
                    resultFile.writelines(str(i) + ",")  # Write each value in the series
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")  # Mark the end of a series
        resultFile.close()  # Close the file after writing
    
        # Write data times (timestamps)
        resultFile = open(os.path.join(imputation_dir,
                                       self.model_name,
                                       self.model_dir,
                                       "batch" + str(batchid) + "delta"), 'w')
        for oneSeries in data_times:
            resultFile.writelines("begin\r\n")  # Mark the beginning of a time series
            for oneClass in oneSeries:
                for i in oneClass:
                    resultFile.writelines(str(i) + ",")  # Write each timestamp
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")  # Mark the end of a time series
        resultFile.close()  # Close the file after writing
    
        # Write the original labels (y)
        resultFile = open(os.path.join(imputation_dir,
                                       self.model_name,
                                       self.model_dir,
                                       "batch" + str(batchid) + "y"), 'w')
        for oneSeries in data_y:
            for oneClass in oneSeries:
                resultFile.writelines(str(oneClass) + ",")  # Write each label
            resultFile.writelines("\r\n")  # New line after each series
        resultFile.close()  # Close the file after writing

    def writeG_Samples(self, filename, step, o):
        # Create a directory to save generated samples if it doesn't exist
        if not os.path.exists(os.path.join("G_results",
                                     self.model_name,
                                     self.model_dir)):
            os.makedirs(os.path.join("G_results",
                                     self.model_name,
                                     self.model_dir))
    
        # Open a file to write generated samples
        resultFile = open(os.path.join("G_results",
                                       self.model_name,
                                       self.model_dir,
                                       filename + str(step)), 'w')
        for oneSeries in o:
            resultFile.writelines("begin\r\n")  # Mark the beginning of a generated series
            for oneClass in oneSeries:
                for i in oneClass.flat:
                    resultFile.writelines(str(i) + ",")  # Write each value in the generated series
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")  # Mark the end of a generated series
        resultFile.close()  # Close the file after writing

    def save(self, checkpoint_dir, step):
        # Create a directory for saving the checkpoint if it doesn't exist
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Save the current session state to the specified checkpoint directory
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re  # Import regex for parsing checkpoint filenames
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir)

        # Check for existing checkpoints in the directory
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restore the model from the checkpoint
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            try:
                reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
                var_to_shape_map = reader.get_variable_to_shape_map()

                available_vars = [
                  v for v in tf.global_variables() if v.name.split(':')[0] in var_to_shape_map
                ]

                saver = tf.train.Saver(available_vars)
                saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

                self.sess.run(tf.global_variables_initializer())
                
                # Extract the counter value from the checkpoint filename
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))  # Confirmation of successful load
                return True, counter  # Return success status and the counter

            except Exception as e:
                print(" [!] Error restoring variables: {}".format(e))
                print(" [*] Proceeding with intialized variables...")
                self.sess.run(tf.global_variables_initializer())
          
        else:
            print(" [*] Failed to find a checkpoint")  # Error message if no checkpoint found
            self.sess.run(tf.global_variables_initializer())
            return False, 0  # Return failure status and zero counter
        
