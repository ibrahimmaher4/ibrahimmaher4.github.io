---
layout: post
title: Machine Learning-Based Leakage Detection For Pipe-In-Pipe System
published: true
---








## Motivation

This Blog Post is to address the challenges of detecting fluid leakage in high risk industries, such as oil and gas, and propose a solution using advanced technologies such as distributed temperature sensing and machine learning. It aim's to provide a more accurate and efficient method for detecting fluid leakage in pipe-in-pipe structures, which can help prevent environmental damage and ensure safety in these industries.









## Method

The method for detecting fluid leakage in a pipe-in-pipe system involves using distributed temperature sensing (DTS) to measure temperature data at various points within the system. Fourier transformed spectrogram data from DTS is then fed into a machine learning algorithm, specifically a convolutional neural network (CNN), which is trained to distinguish between leakage and non-leakage states. The optimized CNN model can detect small amounts of fluid leakage with high accuracy, even in the presence of temperature changes in the working fluid.






## Experiment Apparatus

A 4:1 scale pipe-in-pipe (PIP) prototype system, which is used to simulate fluid leakage. The apparatus includes a metering pump for constant leakage of water, a test section where the leakage is simulated, and a distributed temperature sensing (DTS) system for leakage detection sensors.

 
The test section comprises an inner pipe, an outer pipe, a cartridge heater, a multipoint thermocouple, and a leakage simulator.

### Inner Pipe

The inner pipe contains the working fluid and is made of STS304 with a length of 3 m. The cartridge heater has a power of 5 kW and is inserted at the center of the inner pipe to ensure that the water temperature remains constant. The water temperature is heated to a maximum of 90°C.

### Outer Pipe

In the PIP system leakage is not apparent on the outside because the outer pipe is made of metal.  However, in this experiment the outer pipe is made of acrylic that was transparent, isolates the external environment and inner pipes. The leakage situation can be observed from the outside. 


<div style="display: flex; justify-content: center;">
  <img src="{{ site.baseurl }}/images/experiment%20apparatus.png"/>{{ site.baseurl }}
</div>





## DTS

DTS stands for Distributed Temperature Sensing. It is a technology that allows for the measurement of temperature at various points along a fiber optic cable. This is achieved by sending a laser pulse through the cable and measuring the backscattered light, which is affected by temperature changes along the cable. DTS has advantages over traditional temperature sensors in that it can measure temperature at any point along the cable, is robust against external impact and high temperature, and can be installed relatively easily. DTS is used to measure temperature data within a pipe-in-pipe system for the purpose of detecting fluid leakage.
The DTS system measured to analyze the temperature of three points, the leakage point and the surrounding two points at 0.5 m intervals. The DTS system for this experiment used the TS3000 FIBERPRO model.
the temperature was measured by the amplitude changes of Raman scattering according to the temperature of the reflection point.
 
<div style="display: flex; justify-content: center;">
  <img src="{{ site.baseurl }}/images/DTS.png"/>{{ site.baseurl }}
</div>

The laser pulse irradiates from the input point proceeds through the optical fiber, and the laser pulse backscattered inside the optical fiber is collected through the direction coupler. The measurement location can be determined by the return time of the laser pulse, and the measurement interval is determined according to the data processing time.










## Data Flow


1.	Data Gathered in the Decoupler

2.	Data goes to the Data Acquisition to be pre-processed.
	
    a.	Preprocessing of the data 3 temperatures (x-, x0, x+)
	
    b.	R1(x-/x0) R2(x0/x+)
	
    c.	Standard deviation between R1 & R2

	<div style="display: flex; justify-content: center;">
  		<img src="{{ site.baseurl }}/images/Standard%20deviation.png"/>{{ site.baseurl }}
	</div>

3.	The data converted from standard deviation using FFT.
	
    a.	By dividing the pre-processed data in 300-s

4.	FFT Data converted into 2D spectrograms.

5.	Then CNN 7:3

6.	t-SNE

<div style="display: flex; justify-content: center;">
	<img src="{{ site.baseurl }}/images/Data%20Flow%20DIagram.png"/>{{ site.baseurl }}
</div>

Data is split to 7 to 3, training and testing respectively. Then the 7 is then split to training and validation on this training to prevent overfitting of the parameters which is weights and the biases. While the testing (3) makes sure Hyper parameters (width, depth, batch size of epochs) don’t overfit.

















## CNN

<div style="display: flex; justify-content: center;">
  <img src="{{ site.baseurl }}/images/CNN%20Flow.png"/>{{ site.baseurl }}
</div>
The process of the Convolutional Neural Network (CNN) is as follows:
 1. Input layer: The preprocessed temperature data is fed into the input layer of the CNN as a 2D image. 
2. Convolutional layer: The input data is convolved with a set of learnable filters to extract features from the image. 
3. Activation function: An activation function is applied to the output of the convolutional layer to introduce non-linearity into the model. 
4. Pooling layer/ Trans Layer: The output of the activation function is down sampled using a pooling layer to reduce the dimensionality of the data and make the model more computationally efficient. 
5. Fully connected layer: The output of the pooling layer is flattened and fed into a fully connected layer, which is a traditional neural network layer that connects every neuron in one layer to every neuron in the next layer. 
6. SoftMax layer: The output of the fully connected layer is fed into a SoftMax layer, which normalizes the output into a probability distribution over the two classes (leakage and non-leakage) . 
7. Loss function: The cross-entropy loss function is used to measure the difference between the predicted probability distribution and the actual distribution. 
8. Optimizer: An optimizer is used to minimize the loss function and update the weights of the model. 
9. Training: The CNN is trained on a set of labeled data using backpropagation to adjust the weights of the model. 
10. Testing: CNN is tested on a set of unseen data to evaluate its performance. 
In summary, the CNN used in the experiment is a deep learning model that consists of multiple layers, including convolutional, pooling, fully connected, and softmax layers. The model is trained on preprocessed temperature data to classify the data as leakage or non-leakage with high accuracy.
 

<div style="display: flex; justify-content: center;">
  <img src="{{ site.baseurl }}/images/CNN%20Diagram.png"/>{{ site.baseurl }}
</div>
## Optimizers

Used four different optimizers to optimize the performance of the Convolutional Neural Network (CNN) model. The optimizers used were ADADELTA, ADAGRAD, SGD, and ADAM.

1. ADADELTA: This optimizer is an extension of the Adagrad optimizer that seeks to reduce its aggressive, monotonically decreasing learning rate. It does this by using a moving average of the squared gradient to scale the learning rate. This optimizer is well-suited for large datasets and has been shown to converge faster than other optimizers.

2. ADAGRAD: This optimizer adapts the learning rate of each parameter based on the historical gradient information. It performs well on sparse datasets and is less sensitive to the choice of the initial learning rate.

3. SGD: This optimizer updates the model parameters in the direction of the negative gradient of the loss function with respect to the parameters. It is a simple and computationally efficient optimizer that is widely used in deep learning.

4. ADAM: This optimizer combines the advantages of both AdaGrad and RMSProp optimizers. It uses adaptive learning rates for each parameter and maintains a moving average of the first and second moments of the gradients. This optimizer is computationally efficient and has been shown to converge faster than other optimizers.

In the experiment, the ADAM optimizer was found to perform the best when combined with a learning rate of 0.0075. The choice of optimizer and learning rate can have a significant impact on the performance of the CNN model, and it is important to experiment with different combinations to find the optimal settings for a given dataset.
 

<div style="display: flex; justify-content: center;">
  <img src="{{ site.baseurl }}/images/Optimizers%20Table.png"/>{{ site.baseurl }}
</div>
## Learning rate

The learning rate is a hyperparameter that controls the step size at which the model weights are updated during training. The learning rate was varied for the ADAM optimizer to optimize the performance of the Convolutional Neural Network (CNN) model. The learning rates tested were 0.0001, 0.01, and 0.0075. The results showed that a learning rate of 0.0075 produced the best performance in terms of accuracy and loss.

<div style="display: flex; justify-content: center;">
  <img src="{{ site.baseurl }}/images/Learning%20Rate.png"/>{{ site.baseurl }}
</div>
To determine the best learning rate, training loss and validation loss were compared for different learning rates using the ADAM optimizer. The results showed that the training loss and validation loss tended to decrease as the learning rate increased up to a certain point, after which the loss started to increase again. This is because a high learning rate can cause the model to overshoot the optimal solution and diverge, while a low learning rate can cause the model to converge slowly or get stuck in a local minimum.

The experiment also analyzed the performance of the CNN model under different conditions, such as the number of epochs and the type of optimizer used. The results showed that the ADAM optimizer outperformed the other optimizers tested (ADADELTA, ADAGRAD, and SGD) in terms of accuracy and loss. The experiment also found that the optimal number of epochs for the CNN model was 100.

In addition, the experiment used t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualize the CNN result data and classify the leakage and non-leakage states based on an epoch increase of the ADAM optimizer with 0.0075 learning rate. The visualization showed that the CNN model was able to classify the datasets with increasing epochs.

 Overall, the Experiment demonstrated that the choice of optimizer and learning rate can have a significant impact on the performance of the CNN model. It is important to experiment with different combinations of hyperparameters to find the optimal settings for a given dataset and model architecture.




 


## References

Hayeol Kim a, Jewhan Lee b, Taekyeong Kim a, Seong Jin Park d, Hyungmo Kimc,Im Doo Junga, Advanced thermal fluid leakage detection system with machine learning algorithm for pipe-in-pipe structure (2023), https://doi.org/10.1016/j.csite.2023.102747.
