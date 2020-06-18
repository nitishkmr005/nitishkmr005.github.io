---
title: "Machine Learning Blog: Mathematics Behind Linear Regression, Logistic Regression and Neural Networks"
date: 2020-06-10
tags: [machine learning, data science, neural networks, computer vision]
header:
   image: "/images/image.jpg"
excerpt: "Machine Learning, Data science, Linear Regression, Logistic Regression, Neural Networks"
---
![image](https://user-images.githubusercontent.com/55267125/85016033-97e01300-b186-11ea-8216-24ae05b92543.png)

![image](https://user-images.githubusercontent.com/55267125/85016132-bd6d1c80-b186-11ea-8659-3f15ed441982.png)

![image](https://user-images.githubusercontent.com/55267125/85016535-8a775880-b187-11ea-99f7-f25ecb683cea.png)

![image](https://user-images.githubusercontent.com/55267125/85016592-a418a000-b187-11ea-8cd0-85bff45dbdde.png)

![image](https://user-images.githubusercontent.com/55267125/85016857-31f48b00-b188-11ea-95ff-5b5a65d763b7.png)

## 6) Neural Network

- <ins> Cost Function </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82839205-3d87c580-9eec-11ea-8de3-c7211f4fde36.png)

  Where   
  L = total number of layers in the network  
  Sl =  = number of units (not counting bias unit) in layer l  
  K = number of output units/classes

- <ins> Backpropagation Algorithm </ins>

  ![image](https://user-images.githubusercontent.com/55267125/83003717-54353600-a02c-11ea-880b-02f2419b5db1.png)

## Forward Propagation Matrix Multiplication Example in Neural Network -

 Let's consider input X with 20*20 5000 images

 ![image](https://user-images.githubusercontent.com/55267125/83005450-70d26d80-a02e-11ea-8c28-770d312a82c5.png)

## BackPropagation of Neural Network (Classification Problem) -

  Assumptions - m=1 (Single Sample)  

  Derivation of Gradients -

  ![image](https://user-images.githubusercontent.com/55267125/82945977-324f9b00-9fbb-11ea-8069-205e5c05b6d1.png)
