# Generative-Adversarial-Network for Time Series Imputation - Data Science Capstone
#### Original Work:
title: Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks\
author: Yonghong Luo, Xiangrui Cai, Ying Zhang, Jun Xu and Xiaojie Yuan\
tensorflow version:1.7 python:2.7

#### Project Description: 
This is a research project for a Data Science Capstone course utilizing the work of the identified authors.  The goal is to understand the methods used, add detailed comments to the code, and improve upon anything that allows.  Our hope is that anyone who reviews the code can understand what each section is accomplishing so that it may be easily reproduced.   

# How to Use
The proposed method is a two-stage method. We first train GAN, then we train the input vector of the generator of GAN.

### 1. Dowload Docker Desktop Application for your specific system and install it.
This is required because the code in use for this project contains Tensorflow 1.7 objects which have been replaced since it was written.  There may be other methods that will still allow the use of the older Tensorflow but this is the one we found to work for us.

https://www.docker.com/products/docker-desktop/

### 2. Open Windows Powershell or command line

#### a. Validate Docker is correctly installed

Enter `docker version` in command line/Windows Powershell

![image](https://github.com/user-attachments/assets/2812e749-110d-4c43-9878-c914fe3abda7)

#### b. Pull down the TensorFlow 1.7 image

Enter `docker pull tensorflow/tensorflow:1.7.0 rc1 py3`

![image](https://github.com/user-attachments/assets/9af7e386-cb93-4f6a-a390-2b5b67dd8ec1)

#### c. Create a new container

Enter `docker run d p 888 6 :888 6 name tensorflow container2 tensorflow/tensorflow:1.7.0 rc1 py3`

![image](https://github.com/user-attachments/assets/4fd398d5-12f6-46b0-9866-4f10c55fbff5)

### 3. Download git zip file

![image](https://github.com/user-attachments/assets/1d0ea635-c14c-4d06-8cf4-5efc1455c110)

After the zip file has been downloaded, unzip its contents and store the folder in a location you will know where to access it.

### 4. Open the Docker Desktop Application

Select the `file` tab in the Desktop Application and import your previous download from your local computer into the `mnt` folder

![image](https://github.com/user-attachments/assets/ad9e13fc-61f2-49f0-b318-31d89b4c7469)




### To run the code, go to the Gan_Imputation folder:
 Execute the Physionet_main.py file, then we will get 3 folders named as "checkpoint" (the saved models), G_results (the generated samples), imputation_test_results (the imputed test dataset) and imputation_train_results (the imputed train dataset).
 
### Go to GRUI floder
Execute the Run_GAN_imputed.py file, then one folder-"checkpoint_physionet_imputed" will be created, go to the "checkpoint_physionet_imputed/30_8_128_64_0.001_400_True_True_True_0.15_0.5" folder, find "result" file, the "result" file stands for the mortality prediction results by The RNN classifier trained on the GAN imputed dataset. The first column is epoch, the second column is accuracy and the last column is the AUC score.
### Final result file location
GRUI/max_auc  is the file that record final auc score
# GAN-test

# Credits
Marcus Snell\
Connor Kozlowski\
Conor Aguilar\
Yonghong Luo, Xiangrui Cai, Ying Zhang, Jun Xu and Xiaojie Yuan

# Sources
https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks
