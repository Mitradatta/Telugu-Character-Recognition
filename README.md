# Telugu Handwritten Character Recognition Using Convolutional Neural Networks

## Table of Contents
- [Introduction](#intro)
- [Related Works](https://github.com/Mitradatta/Telugu-Character-Recognition-/edit/main/README.md#related-works)
- [Dataset Description](https://github.com/Mitradatta/Telugu-Character-Recognition-/edit/main/README.md#dataset-description)
- [Image Preprocessing](https://github.com/Mitradatta/Telugu-Character-Recognition-/edit/main/README.md#image-pre-processing)

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
 
 ## Image Pre-processing
 
 ### Image Resizing
 <p align="justify">
 Resizing images is a difficult step in computer vision. Primarily, our machine
learning models train faster if images are smaller. An input image which is two
times as large requires our network to learn from four times as many pixels which
increases the time to generate outputs. Also, many deep learning model
architectures require all images to be in the same size but our raw collected
images may vary in size. So, we resized all our images to a uniform size.
</p>

## Image
