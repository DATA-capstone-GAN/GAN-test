# Processes the imputed results from the WGAN_GRI so it can be fed to the Run_GAN_imputed.py script.
# Finds the epoch with the highest AUC score.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:50:14 2018

@author: luoyonghong
"""
import os                                                                         # Used for handling file and directory paths.
import random                                                                     # Used to shuffle the data for the training process.
class ReadImputedPhysionetData:
    def __init__(self, dataPath ):
        # One file per batch, but need to pay attention to the matching between x, y, and delta
        # Example: batch1y, batch1x, batch1delta
        # Batch ID starts from 1
        self.files = os.listdir(dataPath)                                         # List all of the files in the directory (dataPath).
        self.dataPath=dataPath                                                    # Store the path for later use. 
        self.count=int(len(self.files)/3)                                         # Stores the number of batches.  Within each batch, there are 3 files (batchX: features, batchY: labels, batchDelta: time dependencies).
        
    def load(self):                                                               # Reads all batches into memory.
        count=int(self.count)                                                     # Set count to equal the number of batches in the imputed data.
        self.x=[]                                                                 # Initialize feature variable list.
        self.y=[]                                                                 # Initialize label variable list.
        self.delta=[]                                                             # Initialize time dependency list variable.
        self.x_lengths=[]                                                         # Stores the number of time steps for each batch since this can vary between batches.
        self.m=[]                                                                 # Stores the mask matrix that will identify which values are missing and which are present.
        for i in range(1,count+1):                                                # Iterate over all numbers in count (the number of batches).
            file_x=open(os.path.join(self.dataPath,"batch"+str(i)+"x"))           # Open the features file (x).
            file_y=open(os.path.join(self.dataPath,"batch"+str(i)+"y"))           # Open the labels file (y).
            file_delta=open(os.path.join(self.dataPath,"batch"+str(i)+"delta"))   # Open the time dependencies file (delta).
            this_x,this_lengths=self.readx(file_x)                                # Read in the feature data for specific batch using function defined below (readx()).
            self.x.extend(this_x)                                                 # Add batch feature data to previously initialized (self.x: feature list).
            self.x_lengths.extend(this_lengths)                                   # Add batch feature lengths to previously initialized (self.x_lengths: feature lengths list)
            self.y.extend(self.ready(file_y))                                     # Read in label data for specific batch using function defined below (ready()) and add label data to previously initialized (self.y: label list).
            this_delta,this_m=self.readdelta(file_delta)                          # Read in time dependency data for specific batch using function defined below (readdelta())
            self.delta.extend(this_delta)                                         # Add time dependency data to previously initialized (self.delta: time dependency list).
            self.m.extend(this_m)                                                 # Add mask matrix data for specific batch to previouly initialized (self.m: mask matrix list).
            file_x.close()                                                        # Close the feature data file for the specific batch.
            file_y.close()                                                        # Close the label data file for the specific batch.
            file_delta.close()                                                    # Close the time dependency data file for the specific batch.
        self.maxLength=len(self.x[0])                                             # Sets the max sequence length for the feature data list to that of the first occurance.  Will assume that every patients sequence after this will be equal or less than this value.
        
        
    def readx(self,x):                                                            # Function used above that reads the feature data in each batch.
        this_x=[]                                                                 # Stores the feature values for the batch.
        this_lengths=[]                                                           # Sores the sequence length of the features for each batch.
        count=1                                                                   # Initializes a counter to 1, to be used for reading in lines of feature data file.
        for line in x.readlines():                                                # Iterate over all lines in the feature file of the batch and read them in.
            if count==1:                                                          # Sets conditional for reading in lines.
                words=line.strip().split(",")                                     # Removes the whitespace and splits the feature data file on commas.
                for w in words:                                                   # Iterate over all of the words in the feature data file.
                    if w=='':                                                     # Set conditional for empty string value.
                        continue                                                  # Keep moving if value is empty string.
                    this_lengths.append(int(w))                                   # Add the length of the sequence to this.lengths list and convert to integer type.
            else:
                if "end" in line:                                                 # If end of line is idenfied keep moving.
                    continue
                if "begin" in line:                                               # If a new line is identified.
                    d=[]
                    this_x.append(d)
                else:
                    words=line.strip().split(",")
                    oneclass=[]
                    for w in words:
                        if w=='':
                            continue
                        oneclass.append(float(w))
                    this_x[-1].append(oneclass)
            count+=1
        return this_x,this_lengths
    
    def ready(self,y):
        this_y=[]
        for line in y.readlines():
            d=[]
            words=line.strip().split(",")
            for w in words:
                if w=='':
                    continue
                d.append(int(w))
            this_y.append(d)
        return this_y
    
    def readdelta(self,delta):
        this_delta=[]
        this_m=[]
        for line in delta.readlines():
            if "end" in line:
                continue
            if "begin" in line:
                d=[]
                this_delta.append(d)
                t=[]
                this_m.append(t)
            else:
                words=line.strip().split(",")
                oneclass=[]
                onem=[]
                for i in range(len(words)):
                    w=words[i]
                    if w=='':
                        continue
                    oneclass.append(float(w))
                    if i==0 or float(w) >0:
                        onem.append(1.0)
                    else:
                        onem.append(0.0)
                this_delta[-1].append(oneclass)
                this_m[-1].append(onem)
        return this_delta,this_m
    
    def shuffle(self,batchSize=128,isShuffle=False):
        self.batchSize=batchSize
        if isShuffle:
            c = list(zip(self.x,self.y,self.m,self.delta,self.x_lengths))
            random.shuffle(c)
            self.x,self.y,self.m,self.delta,self.x_lengths=zip(*c)
            
    def nextBatch(self):
        i=1
        while i*self.batchSize<=len(self.x):
            x=[]
            y=[]
            m=[]
            delta=[]
            x_lengths=[]
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                x.append(self.x[j])
                y.append(self.y[j])
                m.append(self.m[j])
                delta.append(self.delta[j])
                x_lengths.append(self.x_lengths[j])
            i+=1
            yield  x,y,[0.0]*len(self.x[0][0]),m,delta,x_lengths,x,0,0,0
#x,y,mean,m,delta,x_lengths,lastvalues
if __name__ == '__main__'     :

    dt=ReadImputedPhysionetData("/Users/luoyonghong/tensorflow_works/Gan_Imputation/imputation_results/35-0.001-1400-18/")
    dt.load()
    print("number of batches is : "+str(dt.count))
    batchCount=1
    for x,y,mean,m,delta,x_lengths,lastvalues,_,_,_ in dt.nextBatch():
        print(batchCount)
        batchCount+=1
