# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:53:29 2018

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
import scipy
from scipy import optimize
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


def Initial_parameters():

    Mean = np.random.randint(0,255, size=(1, 100))
    #print "initial mean:", Mean
    Cov=np.zeros((100,100))
    for j in range(100):
        for k in range(100):
            if j==k:
                Cov[j][k]=int((np.random.random()))
            else:
                Cov[j][k]=0
    Cov = Cov + 1*np.eye(100)
    v = 300
    D = 100
    return  Mean, Cov, v, D


def Estep( flattened_space_face, Mean, Cov, v, D):
    t=len(flattened_space_face)


    expectation_h = np.zeros(t)
    expectation_logh = np.zeros(t)
    d1 = np.zeros(t)

    for i in range(t):
        d = np.matmul((flattened_space_face[i] - Mean),(inv(Cov[:,:])))
        d1[i] = np.matmul(d,((flattened_space_face[i] - Mean).transpose()))
        expectation_h[i]=(v+D)/(v+d1[i])
        expectation_logh[i] = scipy.special.digamma(((v/2) + (D/2)), out=None) - np.log((v/2) + (d1[i]/2))

    return expectation_h, expectation_logh, d1

def Mstep(expectation_h, expectation_logh,  flattened_space_face, v ):
    t=len(flattened_space_face)

    y_sum = np.sum(expectation_h)
  #  print "y_sum", y_sum
    flattened_array = np.asarray(flattened_space_face)
    y_temp = np.zeros((t,100))
    for i in range(t):
        y_temp[i] = expectation_h[i]*flattened_array[i]
 #       print "y_temp", y_temp
    y_temp_sum = np.sum(y_temp, axis=0)
    new_Mean = y_temp_sum/y_sum

    cov_temp = np.zeros((100,100, t))
    for i in range(t):
       cov_temp[:,:,i] = np.matmul(((flattened_array[i]-Mean).transpose()), expectation_h[i]*(flattened_array[i]-Mean))
    new_Cov = (np.sum(cov_temp, axis=2))/y_sum
    new_Cov = np.diag(np.diag(new_Cov))

    def f(v):
        function_v = (t * ((v/2)* log(v/2))) + (t* log(scipy.special.gamma(v/2))) - (((v/2)-1)* np.sum(expectation_logh)) + ((v/2)*np.sum(expectation_h))
        return function_v

    final = scipy.optimize.fmin(f, v)
    new_v = final[0]

    return new_Mean, new_Cov, new_v

def Norm(flattened_space_face, Mean, Cov, v, D):
    d1 = np.zeros(len(flattened_space_face))
    final_test = np.zeros(len(flattened_space_face))
    for i in range(len(flattened_space_face)):
        d = np.matmul((flattened_space_face[i] - Mean),(inv(Cov[:,:])))
        d1[i] = np.matmul(d,((flattened_space_face[i] - Mean).transpose()))
        final_test[i] = (scipy.special.gamma((v+D)/2) * ((1 + (d1[i]/v))**(-(v+D)/2)))/scipy.special.gamma(v/2)
    return final_test

def Posterior(test_face, test_non):
    Posterior_face = test_face/(test_face + test_non)
    Posterior_non = test_non/(test_face + test_non)
    return Posterior_face, Posterior_non
# #############################################################################
# -----------------------------------------------------------------------------

iteration = 7
flattened_space_face = path(1)
flattened_space_non = path(2)
flattened_space_test_face = path(3)
flattened_space_test_non = path(4)

Mean, Cov, v, D = Initial_parameters()
Mean_non, Cov_non, v_non, D_non = Initial_parameters()
print "initial Cov face:", Cov
print "initial Cov non face:", Cov_non

for iter in range(iteration):
    if iter < iteration:
        print "iter: ", iter
        expectation_h, expectation_logh, d1 = Estep(flattened_space_face, Mean, Cov, v, D)
        Mean, Cov, v = Mstep(expectation_h, expectation_logh,  flattened_space_face, v )
        print "v:", v

        expectation_h_non, expectation_logh_non, d1_non = Estep(flattened_space_non, Mean_non, Cov_non, v_non, D_non)
        Mean_non, Cov_non, v_non = Mstep(expectation_h_non, expectation_logh_non,  flattened_space_non, v_non )
        print "non v:", v_non

print "minimum v value", v
print "new Mean", Mean
print "new cov", Cov

print "minimum v non value", v_non
print "new Mean non ", Mean_non
print "new cov non", Cov_non

Mean_face_show=np.reshape(Mean,(10,10))
cv2.imwrite('Mean_face_t-distribution.jpg', Mean_face_show)

Mean_non_face_show=np.reshape(Mean_non,(10,10))
cv2.imwrite('Mean_nonface_t-distribution.jpg', Mean_non_face_show)


cv2.imwrite('Cov_face_tdist_'+str(i)+'.jpg',Cov)

cv2.imwrite('Cov_nonface_tdist_'+str(i)+'.jpg',Cov_non)


# #############################################################################
log_pdf_Testface_wrt_face =  Norm(flattened_space_test_face, Mean, Cov, v, D)
log_pdf_Testface_wrt_non =  Norm(flattened_space_test_face, Mean_non, Cov_non, v_non, D_non)

log_pdf_Testnon_wrt_face =  Norm(flattened_space_test_non, Mean, Cov, v, D)
log_pdf_Testnon_wrt_non =  Norm(flattened_space_test_non, Mean_non, Cov_non, v_non, D_non)


# #############################################################################


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

# ##############################################################################
# PLot ROC Curve

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plt.plot(fpr, tpr, color='darkorange')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.show()
