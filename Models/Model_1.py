# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 02:43:01 2018

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
        imageDir_face = "Face_Classification_1/Dataset/rain_face"
    elif n==2:
        imageDir_face = "Face_Classification_1/Dataset/train_non"
    elif n==3:
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
        image = cv2.imread(imagePath)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flattened = image_gray.flatten()
        flattened_space_face.append(flattened)
    return flattened_space_face

# ##############################################################################
# Calculate Mean and Covariance

def parameters(flattened_space_face):
    Mean = np.mean(flattened_space_face, axis=0)
    Cov = np.cov(np.transpose(flattened_space_face))
    return Mean, Cov

# ##############################################################################
# Calculate Norm (log likelihood)

def Norm(flattened_space_test, Cov, Mean):
    log_pdf = np.zeros(len(flattened_space_test))
    for i in range(len(flattened_space_test)):
        d = np.matmul((flattened_space_test[i] - Mean),(inv(Cov)))
        #print d_face
        d1 = np.matmul(d,((flattened_space_test[i] - Mean).transpose()))
        #print d1_face
        pdf = np.exp(-0.5*d1)
        log_pdf[i] = np.log(pdf)
    print "log_likelihood pdf face", log_pdf
    return log_pdf

# ##############################################################################
# Calculate Posterior

def Posterior(test_face, test_non):
    Posterior_face = test_face/(test_face + test_non)
    Posterior_non = test_non/(test_face + test_non)
    return Posterior_face, Posterior_non

# #############################################################################
# -----------------------------------------------------------------------------

flattened_space_face = path(1)
flattened_space_non = path(2)
flattened_space_test_face = path(3)
flattened_space_test_non = path(4)

Mean_face, Cov_face = parameters(flattened_space_face)
Mean_non, Cov_non = parameters(flattened_space_non)

Mean_show=np.reshape(Mean_face,(10,10))
cv2.imwrite('Mean_single_gaussian_face.jpg', Mean_show)

Mean_show_non=np.reshape(Mean_non,(10,10))
cv2.imwrite('Mean_single_gaussian_nonface.jpg', Mean_show_non)

cv2.imwrite('Cov_single_gaussian_face.jpg', Cov_face)
cv2.imwrite('Cov_single_gaussian_nonface.jpg', Cov_non)

log_pdf_face_wrt_face = Norm(flattened_space_test_face, Cov_face, Mean_face)
log_pdf_non_wrt_face = Norm(flattened_space_test_non, Cov_face, Mean_face)

log_pdf_face_wrt_non = Norm(flattened_space_test_face, Cov_non, Mean_non)
log_pdf_non_wrt_non = Norm(flattened_space_test_non, Cov_non, Mean_non)

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
print "False Positive Rate ", float(count_false_positive)/len(Posterior_testface_face)

count_false_negative = 0
for i in range(len(Posterior_testface_face)):
    if Posterior_testface_non[i]> 0.5:
        count_false_negative=count_false_negative+1
print "False Negative Rate ", float(count_false_negative)/len(Posterior_testface_face)

misclassification = (float(count_false_positive + count_false_negative))/ (len(Posterior_testface_face) + len(Posterior_testface_face))
print "Misclassification Rate", misclassification

Posterior = np.append( Posterior_testface_face, Posterior_testnon_face)
labels = np.append(np.ones(100), np.zeros(100)   )

# ##############################################################################
# PLot ROC Curve

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=0)
plt.plot(fpr, tpr, color='darkorange')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate" )
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.show()

# #############################################################################
# #############################################################################
