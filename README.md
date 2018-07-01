# Face Classification

This repository has been created as an attempt to solve classification of an image into face and non-face using 5 different techniques listed below:

1. Simple Gaussian Model
2. Mixture of Gaussian Model
3. t-distribution Model
4. Mixture of t-distribution Model
5. Factor Analyzer

### Packages and Tools Required:

1. Python 2.7
2. OpenCV 3.1.0
3. NumPy

### Data Preparation:

Data preparation is a very important aspect in computer vision. It starts with data collection followed by cleaning of the data which involves removing garbage values, and then pre process it in such a way so as to be able to use it in any given model.

_You can collect your own dataset from the steps given below or simply use my dataset._

- To collect your own Dataset follow the steps:

 1. __Data Collection:__ I have collected my data from [link](http://vis-www.cs.umass.edu/fddb/).

 2. __Data Cleaning:__ You will need to perform image extraction and cleaning using this README [link](http://vis-www.cs.umass.edu/fddb/README.txt).

- I have already extracted and cleaned the dataset, to use this simply go on this [link](https://drive.google.com/drive/folders/1kn5LzlMARWc0HYY1upxdKZ2ayj_fZ2GI?usp=sharing). Save the 'Dataset' folder inside the main Face_Classification_1 folder. The specifications of my dataset is given below:


  | Folder Name | No. of Images |  Size of Images |
  | ------------| --------------| ----------------|
  |train_face   |     917       | 10x10           |
  | train_non (non face)  | 917 | 10x10|
  |test_face| 100 | 10x10|
  |test_non (non face)| 100| 10x10|


### Running Different Models

  (___Note___: I have converted my dataset into gray scale for the ease of computation. I have, then, flattened every image (shape: 100x1) which can now be used in every model. The code is adaptable to colored images too.)

  1. __Model 1: Simple Gaussian Model: __

    - __Brief Description of the model:__ This model is the simplest of all the models with very less complexity and easy to compute. I have taken my entire training data (consisting of both face and non face folders) and fit two corresponding simple gaussian models for each face class and non face class. Each gaussian has its corresponding mean and covariance matrix. The mean was a 100x1 matrix which was resized back to the initial 10x10 matrix. For the test dataset, the likelihood is calculated for every image in this dataset with respect to both the classes. Next we calculate the Posterior probability given that the prior probability for both classes is same. Finally, we compare the Posterior for one particular image, given both the classes. Applying Bayes Rule, we can find the class to which the given image belonged to.

    - __Guidelines to run the model:__ Run: python Face_Classification_1/Models/Model_1.py

  2. __Model 2: Mixture of Gaussian Model: __

    - __Brief Description of the model:__ This model is an upgrade to simple gaussian model. Here, we divide my two classes (face and non face) to further clusters. Next, we calculate the mean and covariance for every clusters in both classes. The weights for every cluster is modified in every iteration. Expectation-Maximization Algorithm is used here. The posterior is thus calculated with the different clusters and the corresponding final weight. The classification, then, is similar to the previous model.

    - __Guidelines to run the model:__ Run: python Face_Classification_1/Models/Model_2.py

  3. __Model 3: t-distribution Model: __  

    - __Brief Description of the model:__ The ​t-distribution is symmetric and bell-shaped, like the ​ normal distribution​ , but has heavier tails, meaning that it is more prone to producing values that fall far from its mean. Here we have a parameter V which is called the degree of freedom. ​ As the number of degrees of freedom grows, the​ t-distribution approaches the normal distribution with mean 0 and variance 1. ​Here the ‘expectation’ is calculated at every iteration and the term v is updated. This model then follows similar steps as the above Mixture of gaussian model where it classifies the test image (apart from the fact that there are no clusters in the simple t-distribution)

    - __Guidelines to run the model:__ Run: python  Face_Classification_1/Models/Model_3.py

  4. __Model 4: Mixture of t-distribution model:__

    - __Brief Description of the model:__ The mixture of t-distribution considers clusters with their mean and covariance for each cluster for every class. The expectation are updated just exactly like simple t-distributions for every cluster in every class.

    - __Guidelines to run the model:__ Run: python Face_Classification_1/Models/Model_4.py

  5. __Model 5: Factor Analyzer:__

    - __Brief Description of the model:__ Factor analysis provides a compromise in which the covariance matrix is structured so that it contains fewer unknown parameters than the full matrix but more than the diagonal form. One way to think about the covariance of a factor analyzer is that it models part of the high-dimensional space with a full model and mops up remaining variation with a diagonal model.

    - __Guidelines to run the model:__ Run: python Face_Classification_1/Models/Model_5.py
