# Mathematics Behind Linear and Logistic Regressions (DRAFT)

## 1) Hypothesis Function, Cost Function, Gradients, Thetas/Weights for Simple Linear Regression 

- <ins> Hypothesis Function </ins>
  
  ![image](https://user-images.githubusercontent.com/55267125/82761279-2c20b980-9e17-11ea-8a3e-dd216c573d3f.png)

- <ins> Cost Function </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82761913-62603800-9e1b-11ea-8d1e-11645ad38a96.png)

- <ins> Theta - Gradient Descent Algorithm </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82761518-b74e7f00-9e18-11ea-83e7-c1f1b88c35d5.png)

  ![image](https://user-images.githubusercontent.com/55267125/82761422-165fc400-9e18-11ea-9676-de276604b02d.png)

## 2) Hypothesis Function, Cost Function, Gradients, Thetas for Multiple Linear Regression

- <ins> Hypothesis Function </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82761578-2035f700-9e19-11ea-8784-bc197dd8e40d.png)

  Vectorized Form - 

  ![image](https://user-images.githubusercontent.com/55267125/82761585-317f0380-9e19-11ea-9a7e-d17dcb01b820.png)

- <ins> Cost Function in vectorized Form </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82761683-e7e2e880-9e19-11ea-9aa1-f4f2ff057e1e.png)

- <ins> Theta - Gradient Descent Algorithm </ins>

 ![image](https://user-images.githubusercontent.com/55267125/82761614-802c9d80-9e19-11ea-955a-912332541407.png)
 
 ![image](https://user-images.githubusercontent.com/55267125/82761599-5a06fd80-9e19-11ea-986c-3232ea78f6bb.png)

## 3) Hypothesis Function, Cost Function, Gradients, Thetas for Logistic Regression

- <ins> Hypothesis Function in vectorized Form </ins>
 
  ![image](https://user-images.githubusercontent.com/55267125/82761950-96d3f400-9e1b-11ea-9f1f-2ab3d9788ed4.png)

- <ins> Cost Function </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82761789-b6b6e800-9e1a-11ea-8ee3-57d8b423184c.png)

  ![image](https://user-images.githubusercontent.com/55267125/82761824-f7166600-9e1a-11ea-9a7b-450fcbc09777.png)

  Cost Function in Vectorized Form - 
  
  ![image](https://user-images.githubusercontent.com/55267125/82761834-0b5a6300-9e1b-11ea-983a-3cbbc7fb377a.png)

- <ins> Theta - Gradient Descent Algorithm </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82761849-2331e700-9e1b-11ea-9197-07e251ab68fc.png)

  ![image](https://user-images.githubusercontent.com/55267125/82761871-3b096b00-9e1b-11ea-9e05-329c2dddd9a2.png)
  
  ![image](https://user-images.githubusercontent.com/55267125/82761985-e0bcda00-9e1b-11ea-9478-90530e8da169.png)

## 4) Cost Function, Gradients, Thetas for Regularized Linear Regression

- <ins> Cost Function </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82762095-a7d13500-9e1c-11ea-8bdb-83a458666a61.png)

- <ins> Theta - Gradient Descent Algorithm </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82762079-840def00-9e1c-11ea-8240-a910c2b24e3f.png)
  
## 5) Cost Function, Gradients, Thetas for Regularized Logistic Regression

- <ins> Cost Function </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82762120-d7803d00-9e1c-11ea-8bc9-d76ad7fe4f2a.png)

Note: The second term sum means to explicitly exclude the bias term,theta0.

- <ins> Gradients - Gradient Descent Algorithm </ins>

  ![image](https://user-images.githubusercontent.com/55267125/82762143-fb438300-9e1c-11ea-88d6-0def346072fc.png)
  
  ![image](https://user-images.githubusercontent.com/55267125/82762152-0dbdbc80-9e1d-11ea-8a93-3ef77e06c5fd.png)
