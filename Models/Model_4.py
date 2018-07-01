# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 02:39:13 2018

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
import scipy
from scipy import optimize
# #############################################################################
# #############################################################################

def path(n):

    if n == 1:
        imageDir_face = "Face_Classification_1/Dataset/train_face"
    elif n == 2:
        imageDir_face = "Face_Classification_1/Dataset/train_non"
    elif n == 3:
        imageDir_face = "Face_Classification_1/Dataset/test_face"
    else:
        imageDir_face = "Face_Classification_1/Dataset/test_non"

    #imageDir_face = "train_face" #specify your path here
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

# #############################################################################
# #############################################################################

def Initial_parameters(flattened_space_face, clusters):
    lamda = np.zeros(clusters)
    v = np.random.randint(100,200, size=clusters)
    Mean = np.random.randint(0,255, size=(clusters, 100))
    #    print "initial mean:", Mean

    Cov=np.zeros((100,100,clusters))
    for i in range(clusters):
        for j in range(100):
            for k in range(100):
                if j==k:
                    Cov[j][k][i]=int((np.random.random())+(1000))
                else:
                    Cov[j][k][i]=0

    return Mean, Cov, lamda, v

# #############################################################################
# #############################################################################

def Estep( clusters, flattened_space_face, D, Mean, Cov, v):

    t=len(flattened_space_face)
    #flattened_space_face = np.asarray(flattened_space_face)
    expectation_h = np.zeros((t,clusters))
    expectation_logh = np.zeros((t,clusters))
    norm_func = np.zeros((t,clusters))
    responsibility = np.zeros((t,clusters))
    d1 = np.zeros((t, clusters))
    d2 = np.zeros((t, clusters))

    for i in range(t):
        for j in range(clusters):
            d = np.matmul((flattened_space_face[i] - Mean[j,:]),(inv(Cov[:,:,j])))
            d1[i][j] = np.matmul(d,((flattened_space_face[i] - Mean[j,:]).transpose()))
            #print "d1", d1[0][0]
            expectation_h[i][j]=(v[j]+D)/(v[j]+d1[i][j])
            expectation_logh[i][j] = scipy.special.digamma((v[j]/2 + D/2), out=None) - np.log((v[j]/2) + (d1[i][j]/2))
            d2[i][j] = ((1 + (d1[i][j]/v[j]))**(-(v[j]+D)/2))
            norm_func[i][j] = (scipy.special.gamma((v[j]+D)/2) * ((1 + (d1[i][j]/v[j]))**(-(v[j]+D)/2)))/scipy.special.gamma(v[j]/2)
            sum_norm = np.sum(norm_func[i])
        for j in range(clusters):
            responsibility[i][j] = norm_func[i][j]/sum_norm

    return expectation_h, expectation_logh, responsibility

# #############################################################################
# #############################################################################

def Mstep(clusters, expectation_h, expectation_logh, responsibility, flattened_space_face, v, D ):
    t=len(flattened_space_face)
    new_lamda = (np.sum(responsibility, axis=0))/t

    y_sum = np.sum(expectation_h, axis=0)
    flattened_array = np.asarray(flattened_space_face)
    y_temp=np.zeros((t,D,clusters))

    new_Mean = np.zeros((clusters,D))
    for j in range(clusters):
        for i in range(t):
             y_temp[i,:,j] = expectation_h[i,j]*flattened_array[i]
        y_temp_sum = np.sum(y_temp, axis=0)
        new_Mean = y_temp_sum/y_sum[j]
        new_Mean = new_Mean.transpose()

    #
    New_Cov=np.zeros((D,D,clusters))
    for i in range(clusters):
        temp1 = np.zeros((t,D))
        temp2 = np.zeros((t,D))
        temp1_1 = np.zeros((D,1))
        temp2_2 = np.zeros((D,1))
        for j in range(t):
            temp1[j] = (flattened_array[j] - new_Mean[i,:])
            temp2[j] = temp1[j,:]*expectation_h[j][i]
        for k in range(D):
            temp1_1[k] = np.sum(temp1[:,k])
            temp2_2[k] = np.sum(temp2[:,k])
        new_cov_temp=np.matmul(temp1_1,(temp2_2).transpose())
        New_Cov[:,:,i]=new_cov_temp/y_sum[i]

    Cov1=np.zeros((100, 100, clusters))
    for j in range(clusters):
        for i in range(100):
            for k in range(100):
                    Cov1[i][k][j] = (((New_Cov[i][k][j] - np.min(New_Cov[:,:,j]))/ (np.max(New_Cov[:,:,j])-np.min(New_Cov[:,:,j]))))
        Cov1[:,:,j] = Cov1[:,:,j]+(100*np.eye(100))
    new_v=np.zeros(clusters)
    for i in range(clusters):
        def f(v):
            function_v = (t * ((v/2)* log(v/2))) + (t* log(scipy.special.gamma(v/2))) - (((v/2)-1)* np.sum(expectation_logh[:,i])) + ((v/2)*np.sum(expectation_h[:,i]))
            return function_v
        final = scipy.optimize.fmin(f, v[i])
        new_v[i] = final[0]

    lamda = new_lamda
    Mean = new_Mean
    Cov = Cov1

    return lamda, Mean, Cov

# #############################################################################
# #############################################################################


def Norm(flattened_space_face, Cov, Mean, clusters, v):
    D=100
    t=len(flattened_space_face)
    #flattened_space_face = np.asarray(flattened_space_face)
    expectation_h = np.zeros((t,clusters))
    expectation_logh = np.zeros((t,clusters))
    norm_func = np.zeros((t,clusters))
    responsibility = np.zeros((t,clusters))
    d1 = np.zeros((t, clusters))
    d2 = np.zeros((t, clusters))
    pdf =  np.zeros((t, clusters))
    sum_pdf = np.zeros(t)

    for i in range(t):
        for j in range(clusters):
            d = np.matmul((flattened_space_face[i] - Mean[j,:]),(inv(Cov[:,:,j])))
            d1[i][j] = np.matmul(d,((flattened_space_face[i] - Mean[j,:]).transpose()))
            #print "d1", d1[0][0]
            expectation_h[i][j]=(v[j]+D)/(v[j]+d1[i][j])
            expectation_logh[i][j] = scipy.special.digamma((v[j]/2 + D/2), out=None) - np.log((v[j]/2) + (d1[i][j]/2))
            d2[i][j] = ((1 + (d1[i][j]/v[j]))**(-(v[j]+D)/2))
            norm_func[i][j] = (scipy.special.gamma((v[j]+D)/2) * ((1 + (d1[i][j]/v[j]))**(-(v[j]+D)/2)))/scipy.special.gamma(v[j]/2)
            sum_norm = np.sum(norm_func[i])
        for j in range(clusters):
            responsibility[i][j] = norm_func[i][j]/sum_norm
            pdf[i][j] =  lamda[j]*responsibility[i][j]
        sum_pdf[i] = np.sum(pdf[i])


    return sum_pdf


def Posterior(test_face, test_non):
    Posterior_face = test_face/(test_face + test_non)
    Posterior_non = test_non/(test_face + test_non)
    return Posterior_face, Posterior_non
# ################################################
# #############################################################################
# #############################################################################

iteration = 12
clusters = 3
D = 100

flattened_space_face = path(1)
flattened_space_non = path(2)
flattened_space_test_face = path(3)
flattened_space_test_non = path(4)



Mean, Cov, lamda, v  = Initial_parameters(flattened_space_face, clusters)
Mean_non, Cov_non, lamda_non, v_non  = Initial_parameters(flattened_space_non, clusters)

for iter in range(iteration):
    if iter < iteration:
        print "iter: ", iter
        expectation_h, expectation_logh, responsibility = Estep( clusters, flattened_space_face, D, Mean, Cov, v)
        lamda, Mean, Cov = Mstep(clusters, expectation_h, expectation_logh, responsibility, flattened_space_face, v, D)

        expectation_h_non, expectation_logh_non, responsibility_non = Estep( clusters, flattened_space_non, D, Mean_non, Cov_non, v_non)
        lamda_non, Mean_non, Cov_non = Mstep(clusters, expectation_h_non, expectation_logh_non, responsibility_non, flattened_space_face, v_non, D)


for i in range(clusters):
    Mean_show=np.reshape(Mean[i],(10,10))
    cv2.imwrite('Mean_Mix_tDIst_face_'+str(i)+'.jpg',Mean_show)


for i in range(clusters):
    Mean_show_non=np.reshape(Mean_non[i],(10,10))
    cv2.imwrite('Mean_Mix_tDIst_non_'+str(i)+'.jpg',Mean_show_non)

for i in range(clusters):
    cv2.imwrite('Cov_Mix_tDIst_face_'+str(i)+'.jpg',Cov)

for i in range(clusters):
    cv2.imwrite('Cov_Mix_tDIst_non_'+str(i)+'.jpg',Cov_non)


#
log_pdf_Testface_wrt_face =  Norm(flattened_space_test_face, Cov, Mean, clusters, v)
log_pdf_Testface_wrt_non =  Norm(flattened_space_test_face, Cov_non, Mean_non, clusters, v_non)

log_pdf_Testnon_wrt_face =  Norm(flattened_space_test_non, Cov, Mean, clusters, v)
log_pdf_Testnon_wrt_non =  Norm(flattened_space_test_non, Cov_non, Mean_non, clusters, v_non)

#
## #############################################################################
#
#
Posterior_testface_face, Posterior_testface_non  = Posterior(log_pdf_Testface_wrt_face, log_pdf_Testface_wrt_non)
Posterior_testnon_face, Posterior_testnon_non  = Posterior(log_pdf_Testnon_wrt_face, log_pdf_Testnon_wrt_non)

count = 0
for i in range(len(Posterior_testface_face)):
    if Posterior_testface_face[i]> Posterior_testface_non[i]:
        count=count+1
print "count of positively classified faces: ", count


count1 = 0
for i in range(len(Posterior_testface_face)):
    if Posterior_testnon_non[i]> Posterior_testnon_face[i]:
        count1=count1+1
print "count of positively classified non-faces: ", count1


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


fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plt.plot(fpr, tpr, color='darkorange')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.show()
