# Machine Learning folder
You can find many projects from ML group, which I am working on.
(projects ending with - COLAB - means, that you can open them in colab environment and start code on Google's GPU)

MAIN PROJECTS:

## Lenet_MNIST
- Lenet is basic Neural Net consist of two series of CONV => TANH => POOL layer sets followed by a fully-connected layer and softmax output
- The LeNet architecture is a seminal work in the deep learning community, first introduced by LeCun in their 1998 paper, Gradient-Based Learning Applied to Document Recognition. As the name of the paper suggests, the authors’ motivation behind implementing LeNet was primarily for Optical Character Recognition (OCR)
- The LeNet architecture is straightforward and small (in terms of memory footprint), making it perfect for learning the basics of CNNs
- I used it for recognizing handwritten digits

## Parkinson desease
- People with Parkinson's disease have difficulty controlling body movements and the patient gradually loses the ability to perform daily tasks. Currently, Parkinson's disease cannot be cured, but after diagnosis, symptoms of the disease can be effectively alleviated. One method of diagnosis is the so-called geometric test (2017), in which the patient draws spirals or waves. Based on such a test, the doctor can determine whether or not the patient has Parkinson's disease.
- So that the doctor does not have to compare each drawn picture with a picture drawn by a healthy person, it can be automated by computer.
- My program can tell with approximately 83% accuracy whether a patient is healthy or ill. This is how the result looks on several test images
- This is just a small example of how artificial intelligence can help people to be more efficient, faster and make better decisions
- Result (go to the folder for more):

![spiral](Parkinson_desease/spiral.png)

## MiniVGGNet
- VGGNet was first introduced by Simonyan and Zisserman in their 2014 paper "Very Deep Learning Convolutional Neural Networks for Large-Scale Image Recognition". The primary contribution of their work was demonstrating that an architecture with very small (3 × 3) filters can be trained to increasingly higher depths (16-19 layers) and obtain state-of-the-art classification on the challenging ImageNet classification challenge
- VGGNet is unique in that it uses 3 × 3 kernels throughout the entire architecture. The use of these small kernels is arguably what helps VGGNet generalize to classification problems outside what the network was originally trained on
- Usually looks like this:
![VGG_architecture](miniVGG_CIFAR10/VGG_architecture.png)

## Visualize architecture
- I am showing how easy it is to visualize architecture on CNN such as LeNet example

## Pre_trained_CNNs
- showing use of pre-trained CNNs such as VGG16, VGG19, ResNet, InceptionV3, Xception on image classification
- example:

![soccer_ball](pre-trained_CNNs/soccer_ball.png)
