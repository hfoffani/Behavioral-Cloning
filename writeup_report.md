---
output:
  pdf_document: default
  html_document: default
---
#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals or steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/my-convnet-car.jpg "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

The drive.py file was modified to accept a new optional argument `weight`. It allows to use different neural network weights with the same architecture which is useful to evaluate the best number of epochs for the model.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.

The input data pipeline make use of small helper class that makes a pipeline self-describing. In my model pipeline looks like:

```python
def pipeline(input_data):
    return input_data \
            | read_images_and_steer() \
            | add_translated_images(MAXTRANSLATE, replace=False) \
            | add_brightness_images(MAXBRIGHT, replace=False) \
            | flip_images_horizontally() \
            | remove_with_normal(SIGMADELZEROS)
```

This function receives a list of lines as the argument and returns a generator. Each function behind the `|` operator receives a generator and returns another one after applying some transformation. Hence it is possible to process very large sets of images without running into memory problems. This generator can be tested outside Keras as is independent from the framework.

Another advantage is that the generator for Keras can be simplified into a small function:

```python
def keras_generator(input_data, batch_size):
    slice_size = batch_size * 5
    X_batch = np.zeros((batch_size, HEIGHT, WIDTH, CHANNELS))
    y_batch = np.zeros(batch_size)
    i = 0
    while True:
        step = np.random.random_integers(0, len(input_data) / slice_size)
        offset = (step * slice_size) % (len(input_data) - slice_size)
        data_slice = input_data[offset: (offset + slice_size)]
        pipe = pipeline(data_slice)
        for image, steer in pipe:
            X_batch[i] = image
            y_batch[i] = steer
            i += 1
            if i >= batch_size:
                yield X_batch, y_batch
                i = 0
```

This function makes use of the pipeline mentioned before. The purpose of `keras_generator` is to shuffle the input data (the contents of the csv file) randomly, feed it into the `pipeline` where its result are converted into numpy arrays as a generator. Keras will do the rest.


###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I use a model heavily based on NVIDIA End to End Learning for Self-Driving Cars (see https://arxiv.org/abs/1604.07316). It consists of five convolution layers (28@31x98, 36@14x47, 48@5x22, 64@3x20, 64@1x18) followed by three full connected layers of 100, 50 and 20 neurons each and the last output layer. I have changed the last FC layer to use 20 neuros instead of 10. See lines 223 to 245 of model.py

The model uses RELU activations layers after each convolution and He et al. normalization (https://arxiv.org/abs/1502.01852) and ELU activations () for the full connected ones. The data is normalized in the model using a Keras lambda layer (code line 221). 

####2. Attempts to reduce overfitting in the model

I added dropout layers, one after each fully connected layer with a keep probability of 50%. It has been proven that dropout layers enhance the generalization properties of the network to avoid overfitting.

####3. Model parameter tuning

There are several parameters that can be tuned in my solution. These are:

1. The offset correction for the left and right cameras.
1. The maximum translation in pixels for generated sintetyc images.
1. The maximum increase or decrease of the brightness in percentage for generated sintetyc images.
1. The standard deviation around zero of the filter for the steering with angle around zero.

The other hyperparameters are:

1. The initial learning rate.
1. The number of epochs.
1. The batch size.


####4. Appropriate training data

One of the main problem I faced was the lack of appropiate hardware. The keyboard and the joystick I have were not able to drive the simulator satisfactorly. My solution then uses only the data provided by Udacity.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a solution was to comply to some guidelines all along the development that I impose to myself:

1. Have a clear separation of concerns. (preprocess, model, evaluation)
1. Only use the Udacity dataset.
1. Do not modify the image feed part of drive.py

My first approach was to try a simple neural network with one input layer and one output layer so I can test the whole process, from reading the images to writing a model that the simulator can use.

The next step was to try a LeNet model with 5 epochs. It did better than the first model, which is roughly a linear regression, but the car did not advance too much.

Testing the model was done using mean squared error as the loss function in training and validation. The validation set was obtained by separating the 20% of the data set appart. The validation loss was a great indicator if the model performs badly but regrettably it did not give many hints if the model was *great*.

The last model that I implemented was the NVIDIA with some modifications.

To combat the overfitting, I added dropout layers to the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and it showed that a good model was not enough. A lot of augmentation data was needed (details below).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

It consists of five convolution layers (28@31x98, 36@14x47, 48@5x22, 64@3x20, 64@1x18) followed by three full connected layers of 100, 50 and 20 neurons each and the last output layer. I have changed the last FC layer to use 20 neuros instead of 10. See lines 223 to 245 of model.py

I have changed the last FC layer to use 20 neuros instead of 10. See lines 223 to 245 of model.py

The model uses RELU activations layers after each convolution and He et al. normalization (https://arxiv.org/abs/1502.01852) and ELU activations () for the full connected ones. The data is normalized in the model using a Keras lambda layer (code line 221).

Here is a visualization of the architecture

![Model of the neural network][image1]

####3. Creation of the Training Set & Training Process

The training set that I use was the one provided by Udacity.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

