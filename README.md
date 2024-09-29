# Capstone - Generative-Adversarial-Networks
## Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks

 author: Yonghong Luo, Xiangrui Cai, Ying Zhang, Jun Xu and Xiaojie Yuan\
 tensorflow version:1.7 python:2.7

 Project Description: This is a research project for a Data Science Capstone course utilizing the work of the identified authors.  The goal is to understand the methods used and add detailed comments to the code and improve upon anything that allows.  Our hope is that anyone who reviews the code can understand what each section is accomplishing so that it may be easily reproduced.   


## The proposed method is a two-stage method. We first train GAN, then we train the input vector of the generator of GAN.
### To run the code, go to the Gan_Imputation folder:
 Execute the Physionet_main.py file, then we will get 3 folders named as "checkpoint" (the saved models), G_results (the generated samples), imputation_test_results (the imputed test dataset) and imputation_train_results (the imputed train dataset).
 
### Go to GRUI floder
Excute the Run_GAN_imputed.py file, then one floder-"checkpoint_physionet_imputed" will be created, go to the "checkpoint_physionet_imputed/30_8_128_64_0.001_400_True_True_True_0.15_0.5" floder, find "result" file, the "result" file stands for the mortality prediction results by The RNN classifier trained on the GAN imputed dataset. The first column is epoch, the second column is accuracy and the last column is the AUC score.
### Final result file location
GRUI/max_auc  is the file that record final auc score
# GAN-test

# Credits
Marcus Snell\
Connor Kozlowski\
Conor Aguilar\
Yonghong Luo, Xiangrui Cai, Ying Zhang, Jun Xu and Xiaojie Yuan
