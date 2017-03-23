# Behavioral Cloning



## Training data set

### Augmentation and Selection of the data

#### Three cameras
I use all the three cameras to train the data. During the training, the model randomly select a camera image among the three. I added +/- 0.14 of steering values to the cameras. This is determined by fitting a model just using the center camera, and then run a regression to find the most appropriate offset.

#### Flip the images horizontally
I filpped each image horizontally and multiply the steering value by -1 accordingly.


#### Resampling the data

The distribution of steering values is very unbalanced. I used a quick and simple approach to resample the data to make it look more balanced.


### Preparation of the training dataset

I found the quality of training data is very critical to the success of enabling the car to finish the entire lap. With good quality of data, the size of training set can be greatly reduced.

Here are some tips that I found useful:

1. Making the turns as smooth as possible
Our model treat an image/steering pair as an independent event. A zig-zag turn will make the trained network make wrong judgement during the turn. For example, a zig-zag turn may have a moment that steering angle is zero, which gives wrong information during the training. I therefore recommend using a mouse or a joystick intead of using a keyboard, since keyboard strokes makes the left and right turns more discrete.

2. Recovery data
Recovery data is very useful in rescuing the unexpected driving behavior.



## Models

I use the NVidia model to train the data.
The overall modeling flow is illustrated as the following

### Preprocessing






## Results