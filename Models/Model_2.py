# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:06:21 2018

@author: arshita
"""

import cv2
import os, os.path
from random import *
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import math
from pylab import *
from numpy.linalg import inv
from sklearn.metrics import roc_curve

# ##############################################################################
# Read training and test dataset

def path(n):

    if n == 1:
        imageDir_face = "Face_Classification_1/Dataset/train_face"
    elif n==2:
        imageDir_face = "Face_Classification_1/Dataset/train_non"
    elif n==3:
        imageDir_face = "Face_Classification_1/Dataset/test_face"
    else:
        imageDir_face = "Face_Classification_1/Dataset/test_non"

    image_path_list_face = []
    valid_image_extensions = [".jpg"] #specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    image_list_face = []

    for root, dirs, files in os.walk(imageDir_face):
        for file in files:
            with open(os.path.join(root, file), "r") as auto:
                extension = os.path.splitext(file)[1]
                if extension.lower() not in valid_image_extensions:
                    continue
                image_list_face.append(os.path.join(root, file))

    flattened_space_face = []
    for imagePath in image_list_face:
    #            print 'k: ', k
        image = cv2.imread(imagePath)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flattened = image_gray.flatten()
        flattened_space_face.append(flattened)
    return flattened_space_face

# ##############################################################################
# Initial Parameters

def Initial_parameters(N_clusters):

    lamda = np.zeros(N_clusters)
    for i in range(N_clusters):
        lamda[i]=1./N_clusters

    Mean = np.random.randint(0,255, size=(N_clusters, 100))

    Cov=np.zeros((100,100,N_clusters))
    for i in range(N_clusters):
        for j in range(100):
            for k in range(100):
                if j==k:
                    Cov[j][k][i]=int((np.random.random())+(1000*(2)))
                else:
                    Cov[j][k][i]=0

    return lamda, Mean, Cov

# ##############################################################################
# E-M Algorithm

def Estep( N_clusters, flattened_space_face, lamda, Mean, Cov):

    pdf = np.zeros((len(flattened_space_face), N_clusters))
    sum_pdf = np.zeros(len(flattened_space_face))
    for i in range (len(flattened_space_face)):
        for k in range (N_clusters):
            d = np.matmul((flattened_space_face[i] - Mean[k]),(inv(Cov[:,:,k])))
            d1 = np.matmul(d,((flattened_space_face[i] - Mean[k]).transpose()))
            pdf[i][k] =  lamda[k]*exp(-0.5*d1)
        sum_pdf[i] = np.sum(pdf[i])
#    print "pdf", pdf

    r = np.zeros((len(flattened_space_face), N_clusters))
    for i in range(len(flattened_space_face)):
        for j in range(N_clusters):
            r[i][j] = pdf[i][j]/sum_pdf[i]
#    print "r", r

    r_temp = np.zeros(N_clusters)
    for i in range(N_clusters):
        r_temp[i]=np.sum(r[:,i])
#    print "r_temp", r_temp
    return r, r_temp

def Mstep(N_clusters, r, r_temp, flattened_space_face ):

    new_lamda = np.zeros(N_clusters)
    for i in range(N_clusters):
        new_lamda[i] = r_temp[i]/np.sum(r_temp)
#    print "New lamda", new_lamda

    r_t=r.transpose()
    new_mean = np.zeros((N_clusters, 100))
    new_mean_temp = np.zeros((len(flattened_space_face),100))
    for i in range (N_clusters):
        for j in range(len(flattened_space_face)):
            for k in range(100):
                new_mean_temp[j][k] = r_t[i][j]*flattened_space_face[j][k]

        for j in range(100):
           new_mean[i][j] = np.sum(new_mean_temp[:,j])
#    print "New Mean", new_mean

    New_Cov=np.zeros((100,100,N_clusters))
    for i in range(N_clusters):
        temp1 = np.zeros((len(flattened_space_face),100))
        temp2 = np.zeros((len(flattened_space_face),100))
        temp1_1 = np.zeros((100,1))
        temp2_2 = np.zeros((100,1))
        for j in range(len(flattened_space_face)):
            temp1[j] = (flattened_space_face[j] - new_mean[i])
            temp2[j] = temp1[j]*r[j][i]
        for k in range(100):
            temp1_1[k] = np.sum(temp1[:,k])
            temp2_2[k] = np.sum(temp2[:,k])
        new_cov_temp=np.matmul(temp1_1,(temp2_2).transpose())
        New_Cov[:,:,i]=new_cov_temp/np.sum(r_temp[i])
#    print "New_Cov",New_Cov[:,:,i]
    Cov1=np.zeros((100, 100,N_clusters))
    for j in range(N_clusters):
        for i in range(100):
            for k in range(100):
                if i==k:
                    Cov1[i][k][j] = (((New_Cov[i][k][j] - np.min(New_Cov[:,:,j]))/ (np.max(New_Cov[:,:,j])-np.min(New_Cov[:,:,j])))+10*1000)
                else:
                    Cov1[i][k][j] = 0
#    print "Cov1", Cov1
    Mean1=np.zeros((N_clusters, 100))
    for j in range(N_clusters):
        for i in range(100):
            Mean1[j][i] = ((new_mean[j][i] - np.min(new_mean[j,:]))/ (np.max(new_mean[j,:])-np.min(new_mean[j,:])))*255#
#    print "Mean1", Mean1

    lamda = new_lamda
    Mean = Mean1
    Cov = Cov1

    return new_lamda, Mean1, Cov1

# ##############################################################################
# Calculate Norm (log likelihood)

def Norm( N_clusters, flattened_space_test, lamda, Mean, Cov):

    pdf = np.zeros((len(flattened_space_test), N_clusters))
    sum_pdf = np.zeros(len(flattened_space_test))
    for i in range (len(flattened_space_test)):
        for k in range (N_clusters):
            d = np.matmul((flattened_space_test[i] - Mean[k]),(inv(Cov[:,:,k])))
            d1 = np.matmul(d,((flattened_space_test[i] - Mean[k]).transpose()))
            pdf[i][k] =  lamda[k]*exp(-0.5*d1)
        sum_pdf[i] = np.sum(pdf[i])
    return  sum_pdf

def Posterior(test_face, test_non):
    Posterior_face = test_face/(test_face + test_non)
    Posterior_non = test_non/(test_face + test_non)
    return Posterior_face, Posterior_non


# #############################################################################
# -----------------------------------------------------------------------------

N_clusters = 3
iteration = 10
flattened_space_face = path(1)
flattened_space_non = path(2)
flattened_space_test_face = path(3)
flattened_space_test_non = path(4)

lamda_face, Mean_face, Cov_face = Initial_parameters(N_clusters)
lamda_non, Mean_non, Cov_non = Initial_parameters(N_clusters)


for iter in range(iteration):
    if iter < iteration:
        print "iter: ", iter
        r, r_temp = Estep( N_clusters, flattened_space_face, lamda_face, Mean_face, Cov_face)
        lamda_face, Mean_face, Cov_face = Mstep(N_clusters, r, r_temp, flattened_space_face )

        r_non, r_non_temp = Estep( N_clusters, flattened_space_non, lamda_non, Mean_non, Cov_non)
        lamda_non, Mean_non, Cov_non = Mstep(N_clusters, r_non, r_non_temp, flattened_space_non )

print "mean:", Mean_face

for i in range(N_clusters):
    Mean_show=np.reshape(Mean_face[i],(10,10))
    cv2.imwrite('Mean_MixtureGaussian_face_'+str(i)+'.jpg',Mean_show)

for i in range(N_clusters):
    Mean_show_non=np.reshape(Mean_non[i],(10,10))
    cv2.imwrite('Mean_MixtureGaussian_non_'+str(i)+'.jpg',Mean_show_non)

for i in range(N_clusters):
    cv2.imwrite('Cov_MixtureGaussian_face_'+str(i)+'.jpg',Cov_face)

for i in range(N_clusters):
    cv2.imwrite('Cov_MixtureGaussian_non_'+str(i)+'.jpg',Cov_non)


log_pdf_face_wrt_face = Norm(N_clusters, flattened_space_test_face, lamda_face, Mean_face, Cov_face)
log_pdf_non_wrt_face = Norm(N_clusters, flattened_space_test_non, lamda_face, Mean_face, Cov_face)

log_pdf_face_wrt_non = Norm(N_clusters, flattened_space_test_face, lamda_non, Mean_non, Cov_non)
log_pdf_face_wrt_face = Norm(N_clusters, flattened_space_test_non, lamda_non, Mean_non, Cov_non)


Posterior_testface_face, Posterior_testface_non  = Posterior(log_pdf_face_wrt_face, log_pdf_face_wrt_non)
Posterior_testnon_face, Posterior_testnon_non  = Posterior(log_pdf_non_wrt_face, log_pdf_non_wrt_non)

count = 0
for i in range(len(Posterior_testface_face)):
    if Posterior_testface_face[i]> Posterior_testface_non[i]:
        count=count+1
#print "count: ", count

count1 = 0
for i in range(len(Posterior_testface_face)):
    if Posterior_testnon_non[i]> Posterior_testnon_face[i]:
        count1=count1+1
#print "count1: ", count1

count_false_positive = 0
for i in range(len(Posterior_testface_face)):
    if Posterior_testnon_face[i]> 0.5:
        count_false_positive=count_false_positive+1
print "False Positive Rate ", float(count_false_positive)/100

count_false_negative = 0
for i in range(len(Posterior_testface_face)):
    if Posterior_testface_non[i]> 0.5:
        count_false_negative = count_false_negative + 1
print "False Negative Rate ", float(count_false_negative)/100

misclassification = (float(count_false_positive + count_false_negative)) / (len(Posterior_testface_face) + len(Posterior_testface_face))
print "Misclassification Rate", misclassification

Posterior = np.append(   Posterior_testface_face, (Posterior_testnon_face))
labels = np.append(np.ones(100), np.zeros(100)   )

# ##############################################################################
# PLot ROC Curve

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plt.plot(fpr, tpr, color='darkorange')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.show()
