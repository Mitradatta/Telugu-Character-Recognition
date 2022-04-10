# Telugu Handwritten Character Recognition Using Convolutional Neural Networks

## Table of Contents
- [Introduction](#intro)
- [Related Works](#works)
- [Dataset Description](#description)
- [Image Preprocessing](#preprocessing)
   - [Image Resizing](#resize)
   - [Gray scale Image converstion](#gray)
   - [Binarization using adaptive Thresholding](#binary)
   - [Median Filtering for Noise Removal](#median)
   - [Data Augmentation](#augmentation)
   - [Normalization](#normalize)
- [Methodology Used](#methodology)
- [Overview of the Convolutional Neural Networks and its Architecture](#overview)
- [CNN Architecture Used](#architecture)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

<span id="intro"> </span>
## Introduction
<p  align = "justify">
Telugu is a prominent south Indian language spoken by about 74 million people. It has 16 vowels (which are also known as achulu) and 36 consonants (which are
also known as hallulu). A single character or an Akshara in an Indian language typically represents an entire syllable, or the coda of one syllable and the onset of
another. Dataset plays an important role in the design of supervised learning systems. Since the datasets available for Telugu characters is very limited and not
available as open source, we have created our own dataset, by collecting samples from different aged people consisting of about 250 samples for each character.
Then we have pre-processed the collected data to make it applicable for our research. In this project we have used convolutional neural networks to identify the hand
written characters. Convolutional Neural Networks (CNNs) have revolutionized the computer vision and pattern recognition, and mainly for offline handwriting
recognition. Neural networks work similar to our neurons in our brain. The main advantage by using this neural network is that the model can learn from the
previous learned data and future predicted data. It can learn while working or predicting and it also has many hidden layers which makes it more accurate.
Another main advantage is that it can automatically select required features, hence feature selection is done by the model itself.
</p>
<span id="works"> </span>

## Related Works
<p align= "justify">
During the last few decades researchers were focused on Support Vector Machine (SVM), Neural Networks (NN), k-Nearest Neighbor (k-NN) for classification. 
In one of the papers, they proposed a hybrid approach for recognition of handwritten Telugu CV type characters, as offline images, using a combination of 
 CNN, PCA, SVM and multi classifier system techniques. The best individual classifier trained on MNIST data yielded a test performance of 98.25% while the ensemble 
 classifier for the same data achieved a test performance of 98.5%. In one of the papers written by Riya Guha, they proposed a new CNN architecture called Dev-net 
 architecture to recognize Hindi characters. It is a 6-layer CNN architecture which requires less time and memory space as compared to the existing CNN models. 
 It has achieved 99.6% accuracy. It also performed better than already existing CNN architectures like LeNet, ResNet, AlexNet etc
The architecture of the proposed recognition scheme for palm leaf scribed text. It is divided into three major phases: 3D data acquisition, Pre-processing, and
Character recognition. In a paper written by K.C. Movva they proposed that the usage of distributions of few properties of stroke points like local variance 
 and local moments can be used to represent a stroke and also suggests some features to improve the shape recognition scheme. A stroke could be viewed as a sequence
of points from pen-up to pen-down. Each Telugu character is comprised of some strokes and thus individual stroke identification and classification has led them to individual 
character classification. In this paper a K-NN and A-NN based models where compared. They found that the artificial neural network (ANN) model has been proven to give good classification 
accuracy even when they use few samples per character which makes it efficient and suitable for handwritten character recognition.
</p>

<span id="description"> </span>
## Dataset Description

<p align="center">
<img width="220" alt="Screenshot 2022-04-09 at 9 10 19 AM" src="https://user-images.githubusercontent.com/54971204/162554967-8665bb1f-838e-4f08-b319-141e22220fa0.png">
</p>

<p align ="justify">
 
We have created our own dataset since the datasets available for Telugu characters is very limited. We have collected data from people belonging to different ages
varying from 8 - 78 years. The dataset contains 250 samples for each character. Each sample contains all the 52 alphabet (Varnamala) that is
16 vowels (Aachulu(a)) and 36 consonants (Hallulu(b)). Initially we have worked only on alphabet of Telugu. Based on the output of the
model, we can improve our dataset by including vothulu and matralu which are extensions to Telugu alphabet. In future, we can extend this
algorithm from character recognition to text ecognition by creating our own dataset.  
 </p>
 
 <span id="preprocessing"> </span>
 ## Image Pre-processing
 
 <span id="resize"> </span>
 ### Image Resizing
 <p align="justify">
 Resizing images is a difficult step in computer vision. Primarily, our machine
learning models train faster if images are smaller. An input image which is two
times as large requires our network to learn from four times as many pixels which
increases the time to generate outputs. Also, many deep learning model
architectures require all images to be in the same size but our raw collected
images may vary in size. So, we resized all our images to a uniform size.
</p>


<p align="center">
 <img width="400" alt="Screenshot 2022-04-10 at 2 28 55 PM" src="https://user-images.githubusercontent.com/54971204/162610907-5f104522-5ff9-4fd7-95ad-e7182a849159.png">
</p>






<span id="gray"> </span>
### Gray scale Image converstion

<p align="justify">
Grayscale image means that the value of each pixel in the image represents only
the intensity information of the light. Grayscale images are only composed of
different shades of gray ranging from black to white (weakest intensity to
strongest intensity).Grayscale image conversion is an Instinctive way to convert a coloured image 3D array to a grayscale 2D array is, for each pixel we will take the average of the red, green, and blue pixel values of the image to get the grayscale value. This combines the lightness contributed by each colour band in the image into a reasonable gray approximation. We do Grayscale image conversion because we can only specify a single intensity value for each pixel, as opposed to the three intensities needed to specify each
pixel in a full colour image. The main reason for differentiating the images from
any other sort of colour image is that only less information needs to be provided
for each pixel so that there will be no need to use more complicated and harder-to-process colour images.
 </p>
 
<p align="center">
<img width="300" alt="Screenshot 2022-04-10 at 2 29 10 PM" src="https://user-images.githubusercontent.com/54971204/162610992-3e060265-7fa1-4040-aca5-d664f6d777ed.png">
</p>

<span id="binary"> </span>
### Binarization using adaptive Thresholding
<p align="justify">
 Binarization is the process of converting any gray – scale image into black – white
image two tone image. To perform binarization, first we need the threshold value
of gray scale and check if a pixel is having a particular gray value or not.
In case the gray value of the pixels is greater than the threshold value, then those
pixels are converted to the white. Similarly, if the gray value of the pixels is less
than the threshold, then those pixels are converted into the black. Normally, we
 find the global threshold for the whole image and binarize the image using a
single threshold value.
 </p>
 
 
 <p align="center">
 <img width="300" alt="Screenshot 2022-04-10 at 2 29 23 PM" src="https://user-images.githubusercontent.com/54971204/162611021-3d8139c3-59ca-4c89-af69-40961548e137.png">
</p>
 
 <span id="median"> </span>
 ### Median Filtering for Noise Removal
 <p align= "justify">
The Median filter is a non-linear digital image filtering method, it is used widely in
removing noise. This noise reduction is an important pre-processing step to
improve the results of future processing. Under certain conditions, Median filter
preserves edges at same time as removing noise.
Median filter is a sliding window that replaces the centre value with the Median
of all the pixel values in the window. The window or kernel is normally a square
but it can be of any shape.
 </p>
 
 <p align="center">
 <img width="300" alt="Screenshot 2022-04-10 at 2 29 36 PM" src="https://user-images.githubusercontent.com/54971204/162611051-19f78aa9-f8a1-4292-8dfb-8e16399cf62d.png">
</p>
 
 <span id="augmentation"> </span>
 ### Data Augmentation
 <p align = "justify">
 Data Augmentation is a popular Machine Learning technique used in making
robust ML models even when the available data is very less. It helps in increasing
the amount of original data by adding slightly modified copies of already existing
data or newly created data from the existing data. Adding a variety of data greatly
helps in reducing over fitting when training a machine learning model on low
quality and sized data. It is observed to work pretty well in computer vision
applications such as Image classification, Object detection etc where we now have
set of transformation function. Data augmentation techniques such as cropping,
padding, and horizontal flipping are most widely used to train large neural
networks.After applying augmentation on our data set we were able to produce 9 images
for each image in our original dataset.
 </p>
 
 <p align="center">
 <img width="500" alt="Screenshot 2022-04-10 at 2 29 56 PM" src="https://user-images.githubusercontent.com/54971204/162611076-2d8bcefd-9e82-40d1-9084-4dfea2881fdb.png">
 </p>
 
 <span id="normalize"> </span>
 ### Normalization
 <p align="justify">
Normalization is a technique mostly applied as part of data preparation in
machine learning. The main aim of normalization is changing the values of
numeral columns in the dataset into a common scale, without effecting
differences in the ranges of values or losing information. Normalization is also
required when using some algorithms to model the data correctly.
 </p>
 
 <p align = "center">
 <img width="300" alt="Screenshot 2022-04-09 at 5 53 20 PM" src="https://user-images.githubusercontent.com/54971204/162574014-a66ee105-812d-4cef-b01d-c7d077276ff6.png">
</p>
Here, max(x) and min(x) are the maximum and the minimum values of the feature used.

 <span id="methodology"> </span>
## Methodology Used
<p align = "justify">
 In spite of being popular, ANNs were unable to handle large dataset in
recognition/classification tasks. To overcome these, a new machine learning
paradigm, deep learning, was introduced. It is a stacked neural network that is
composed of several layers. Earlier versions of neural networks, such as the first
perceptron’s were shallow, composed of one input and one output layer; and one
hidden layer in between. In deep-learning networks, each layer of nodes trains on
a distinct set of features based on the previous layer’s output. A model will be
efficient if hidden layers have the ability to learn complicated features from
observed data. Deep neural network shows notable performance on unseen data.
Some popular deep neural network architectures are recurrent neural networks
(RNNs), CNNs, deep belief networks, auto-encoders and generative adversarial
networks. In general, CNNs are considered as a machine learning architecture,
which has a capability to learn from experiences like multilayer neural network
with back propagation. For the requirement of minimal pre-processing, CNNs use
a variation of the multilayer perceptron. CNNs are composed of an automatic
feature extractor and a trainable classifier, having important layers
 </p>
 
- Convolutional Layer (CL)
- Pooling Layer (PL)
- Fully-Connected Layer(FCL)

<span id="overview"> </span>
## Overview of the Convolutional Neural Networks and its Architecture
<p align="justify">
Basic CNN models use CLs and PLs and provide a standard architecture. In CNN’s a
series of convolution operation along with pooling and non-linearity activation
function are applied to the input and passing the result to the next layer. The
filters (F) are applied in the CL to extract relevant features from the input image
to pass further. Each filter gives a different feature for correct prediction. To
retain the size of the image, same padding (zero padding) is applied, otherwise
valid padding is used, since it helps reduce the number of features. The
convoluted output is obtained as an activation map.
 </p>
 
Input size=MxN

P=(F-1)/2

(M+2P x N+2P)\*(FxF)

P-Padding

F -is the size of the filter used (3x3)

Output=MxN

<span id="architecture"> </span>
## CNN Architecture Used:
We provided a basic overview of CNN architecture in the above section. We
proposed a CNN architecture that is designed to recognize telugu characters.
The architecture in the below diagram, which comprises of 6 layers, excluding input. The input image is a 76x80x1 pixel image. Firstly, the size of the input image is resized to (76x80). Then the first layer takes image pixels as input. Each convolution layers are in an alternating position with sub-sampling or pooling layers (Max pooling size-2x2), which take the pooled maps as input. In proposed model CLs have (3x3) Filter with same Padding. All the convolution stride is fixed to one. For every Convolution layer we have used RELU activation function. The final dense output was classified using soft max function.

<p align="center">
 
 <img width="1234" alt="Screenshot 2022-04-10 at 2 30 53 PM" src="https://user-images.githubusercontent.com/54971204/162611441-58a68cd3-901b-47e3-bc35-db79b6151658.png">
 </p>
 
 
 <span id="results"> </span>
 ## Results
 <p align="justify">
 All the experiments are performed over a system having specifications as MacOS,
64 bit operating system , and Intel(R) i5 @1.5 GHz dual core and all the
stimulations has been done through Jupyter notebook over a standard dataset.
The standard dataset contains 250 samples from each of the 52 categories Telugu
character Dataset. The accuracy of the model for the training dataset has shown
98% and the accuracy of the trained model for testing data was 94.6%.
 </p>
 
 
<span id="conclusion"> </span> 
 ## Conclusion
 <p align="justify">
 We presented a comprehensive and practical CNN system for Telugu Language.
The proposed system is shape and edge dependent and requires pre-processing
and feature extraction. The experimental result shows, the performance
characteristics of the Convolution Neural Network. To conclude, the design,
approach and implementation were driven by the need for a practical CNN
system for Telugu handwritten character recognition. Based on the output of the
model, we would like to improve our dataset by including vothulu and matralu
which are extensions to Telugu alphabet. In future, we can extend this algorithm
from character recognition to text recognition by creating our own dataset.
  </p>
  
 
<span id="references"> </span> 
  ## References
- https://www.researchgate.net/publication/328745907_A_MomentBased_Representation_for_Online_Telugu_Handwritten_Character_Recognition_Proceedings_of_DAL_2018
- https://www.worldscientific.com/doi/epdf/10.1142/S0218001420520096
- S. T. Soman, A. Nandigam and V. S. Chakravarthy, "An efficient multiclassifier system based on convolutional neural network for offline handwritten Telugu character recognition," 2013 National Conference on Communications (NCC), 2013, pp. 1-5, doi: 10.1109/NCC.2013.6488008.
- https://www.researchgate.net/publication/304576753_A_novel_3D_approach_to_recognize_Telugu_palm_leaf_text
- https://www.researchgate.net/publication/334488105_UHTelPCC_A_Dataset_for_Telugu_Printed_Character_Recognition
