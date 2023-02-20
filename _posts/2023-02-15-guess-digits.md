---
layout: post
title: Guessing user drawn digit
date:   2023-02-15 12:00:00 +0700
tags: keras streamlit
featured_img: /assets/images/posts/guess-digit/demo.gif

---
In this post I made a fun game of drawing a number, and challenging the model to guess it correctly. The model itself is a Convolutional Neural Network (CNN) model, built using the MNIST digits datasets.


Here I want to combine it with a streamlit apps that can take live user inpu, making the inference process more interactive.




[Try on Streamlit](https://dobeok-guess-digit-app-app-aggm7n.streamlit.app/)


<video controls autoplay height="480">
  <source src="/assets/images/posts/guess-digit/guess-digit-demo.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>


## I. Streamlit code (front-end)

The full code can be found in [my repo](https://github.com/dobeok/guess-digit-app). Here's a minimal code to highlight the main logic.

<script src="https://gist.github.com/dobeok/953becc81efecf5d8f1554e6fdd1c103.js"></script>

## II.Model building code (back-end)

### 0. Imports

<script src="https://gist.github.com/dobeok/bfd56bdcfe4230218129beb5c6e45082.js"></script>


### 1. Load & explore data

<script src="https://gist.github.com/dobeok/209519729f6b1353684d4905e0448543.js"></script>


#### 1.1 Sample different ways to write each number

<script src="https://gist.github.com/dobeok/2cb2abed013652802bf4f748bd38613c.js"></script>

![png](/assets/images/posts/guess-digit/model_files/model_sample_train_data.png)


#### 1.2 Check if classes are balanced

<script src="https://gist.github.com/dobeok/3a44b2f3248d545388a315a1013e9a67.js"></script>

    
![png](/assets/images/posts/guess-digit/model_files/model_7_1.png)
    

### 2 Transform data

<script src="https://gist.github.com/dobeok/962492ac801daa93b47743d33797b329.js"></script>


### 3. Model


<script src="https://gist.github.com/dobeok/4d988f917b3ff697386510a755dd64bb.js"></script>

```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 1600)              0         
                                                                     
     dropout (Dropout)           (None, 1600)              0         
                                                                     
     dense (Dense)               (None, 10)                16010     
                                                                     
    =================================================================
    Total params: 34,826
    Trainable params: 34,826
    Non-trainable params: 0
    _________________________________________________________________
```

<script src="https://gist.github.com/dobeok/9a482db762760093a8a2d40ec85a52cf.js"></script>

```
    Epoch 1/15


    2023-02-19 22:15:15.668285: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
    2023-02-19 22:15:15.874289: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    422/422 [==============================] - ETA: 0s - loss: 0.3702 - accuracy: 0.8868

    2023-02-19 22:15:23.874575: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    422/422 [==============================] - 9s 17ms/step - loss: 0.3702 - accuracy: 0.8868 - val_loss: 0.0942 - val_accuracy: 0.9743
    Epoch 2/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.1146 - accuracy: 0.9653 - val_loss: 0.0643 - val_accuracy: 0.9823
    Epoch 3/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0856 - accuracy: 0.9739 - val_loss: 0.0514 - val_accuracy: 0.9865
    Epoch 4/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0716 - accuracy: 0.9789 - val_loss: 0.0483 - val_accuracy: 0.9872
    Epoch 5/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0639 - accuracy: 0.9797 - val_loss: 0.0446 - val_accuracy: 0.9868
    Epoch 6/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0573 - accuracy: 0.9829 - val_loss: 0.0391 - val_accuracy: 0.9902
    Epoch 7/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0508 - accuracy: 0.9841 - val_loss: 0.0368 - val_accuracy: 0.9907
    Epoch 8/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0484 - accuracy: 0.9844 - val_loss: 0.0358 - val_accuracy: 0.9890
    Epoch 9/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0441 - accuracy: 0.9869 - val_loss: 0.0369 - val_accuracy: 0.9897
    Epoch 10/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0413 - accuracy: 0.9871 - val_loss: 0.0344 - val_accuracy: 0.9898
    Epoch 11/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0390 - accuracy: 0.9874 - val_loss: 0.0345 - val_accuracy: 0.9907
    Epoch 12/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0359 - accuracy: 0.9886 - val_loss: 0.0335 - val_accuracy: 0.9905
    Epoch 13/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0361 - accuracy: 0.9886 - val_loss: 0.0350 - val_accuracy: 0.9895
    Epoch 14/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0323 - accuracy: 0.9889 - val_loss: 0.0314 - val_accuracy: 0.9913
    Epoch 15/15
    422/422 [==============================] - 6s 15ms/step - loss: 0.0320 - accuracy: 0.9895 - val_loss: 0.0299 - val_accuracy: 0.9917
```


<script src="https://gist.github.com/dobeok/d270ab600266430703e6e39e537d7e8a.js"></script>

    
![png](/assets/images/posts/guess-digit/model_files/model_21_0.png)
    

### 4. Evaluate performance


#### 4.1 Predict on test data

<script src="https://gist.github.com/dobeok/e87ae62735b7cb6676f35fbbd8d30524.js"></script>


#### 4.2 Confusion matrix

<script src="https://gist.github.com/dobeok/6a83910d6aed3a30b0b17412ac8a7b10.js"></script>
    
![png](/assets/images/posts/guess-digit/model_files/model_33_1.png)
    


Observations:
- Majority of predictions are on the correct diagonal
- 3 and 5; 4 and 9 are commonly mistaken pairs, which is understandable because even human might make the same mistakes

### 5. What's next?

* Augment data. The images in the tranining datasets are quite small (28 x 28 pixels) so I had to downsize the input image. Otherwise the drawing canvas would be very small. There are some augmentation techniques that I think will improve the model:
  * Translation
  * Rotation
  * Scaling