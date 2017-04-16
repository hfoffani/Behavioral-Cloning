# Behavioral Cloning


The goals or steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/my-convnet-car.jpg "Model Visualization"
[image2]: ./examples/center_037.jpg "Center Camera"
[image3]: ./examples/left_037.jpg "Left Camera"
[image4]: ./examples/right_037.jpg "Right Camera"
[image5]: ./examples/trans-00008.jpg "Translated Image 1"
[image6]: ./examples/trans-0024.jpg "Translated Image 2"
[image7]: ./examples/trans-0031.jpg "Translated Image 3"
[image8]: ./examples/bright-0044.jpg "Brightness Image 1"
[image9]: ./examples/bright-0045.jpg "Brightness Image 2"
[image10]: ./examples/flip-0013.jpg "Flipped Image 1"
[image11]: ./examples/flip-0020.jpg "Flipped Image 2"


My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

The drive.py file was modified to accept a new optional argument `weight`. It allows to use different neural network weights with the same architecture which is useful to evaluate the best number of epochs for the model.

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


### Model Architecture and Training Strategy

I use a model heavily based on NVIDIA End to End Learning for Self-Driving Cars (see https://arxiv.org/abs/1604.07316). It consists of five convolution layers (28@31x98, 36@14x47, 48@5x22, 64@3x20, 64@1x18) followed by three full connected layers of 100, 50 and 20 neurons each and the last output layer. I have changed the last FC layer to use 20 neurons instead of 10. See lines 223 to 245 of model.py

The model uses RELU activation layers after each convolution and He et al. normalization (https://arxiv.org/abs/1502.01852) and ELU activations () for the full connected ones. The data is normalized in the model using a Keras lambda layer (code line 221). 


I added dropout layers, one after each fully connected layer with a keep probability of 50%. It has been proven that dropout layers enhance the generalization properties of the network to avoid over-fitting.


There are several parameters that can be tuned in my solution. These are:

1. The offset correction for the left and right cameras.
1. The maximum translation in pixels for generated synthetic images.
1. The maximum increase or decrease of the brightness in percentage for generated intensity images.
1. The standard deviation around zero of the filter for the steering with angle around zero.

The other hyper-parameters are:

1. The initial learning rate.
1. The number of epochs.
1. The batch size.


One of the main problem I faced was the lack of appropriate hardware. The keyboard and the joystick I have were not able to drive the simulator satisfactorily. My solution then uses only the data provided by Udacity.


The overall strategy for deriving a solution was to comply to some guidelines all along the development that I impose to myself:

1. Have a clear separation of concerns. (preprocess, model, evaluation)
1. Only use the Udacity data-set.
1. Do not modify the image feed part of drive.py

My first approach was to try a simple neural network with one input layer and one output layer so I can test the whole process, from reading the images to writing a model that the simulator can use.

The next step was to try a LeNet model with 5 epochs. It did better than the first model, which is roughly a linear regression, but the car did not advance too much.

Testing the model was done using mean squared error as the loss function in training and validation. The validation set was obtained by separating the 20% of the data set apart. The validation loss was a great indicator if the model performs badly but regrettably it did not give many hints if the model was *great*.

The last model that I implemented was the NVIDIA with some modifications.

To combat the over-fitting, I added dropout layers to the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and it showed that a good model was not enough. A lot of augmentation data was needed (details below).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

It consists of five convolution layers (28@31x98, 36@14x47, 48@5x22, 64@3x20, 64@1x18) followed by three full connected layers of 100, 50 and 20 neurons each and the last output layer. I have changed the last FC layer to use 20 neurons instead of 10. See lines 223 to 245 of model.py

I have changed the last FC layer to use 20 neurons instead of 10. See lines 223 to 245 of model.py

The model uses RELU activations layers after each convolution and He et al. normalization (https://arxiv.org/abs/1502.01852) and ELU activations () for the full connected ones. The data is normalized in the model using a Keras lambda layer (code line 221).

Here is a visualization of the architecture

![Model of the neural network][image1]


#### Creation of the Training Set & Training Process

The training set that I use was the one provided by Udacity.

I took the images first from the center camera only. I then added the images from the left and right cameras and applied a correction to the steer angle. The next figure shows the images from the center, left and right camera respectively.

![Center Camera][image2]
![Left Camera][image3]
![Right Camera][image4]

Still this data set was not able to drive the simulator correctly without sliding of the lane.

But a substantial increase of the training data through augmentation did the trick. I added an horizontal translation with a random move to the left or right of up to 50 pixels per side.

![Translated Image 1][image5]
![Translated Image 2][image6]
![Translated Image 3][image7]

I also added a random brightness increase or decrease of up to 50%.

![Brightness Image 1][image8]
![Brightness Image 2][image9]

For each of all the generated images I also added its corresponding flipped image so the network would not learn that the circuit is counterclockwise and prefer to infer left turns.

![Flipped Image 1][image10]
![Flipped Image 2][image11]

All these steps are joined together with the aforementioned pipeline. It roughly generates around 50,000 images. However the distribution of this set shows that there are thousands of images that correspond to driving straight or with a very small steer angle. Without filtering these observations the network would tend to drive straight failing to take the curves. The last step of the pipeline filter these images with a Gaussian probability (i.e. the probability to be filter out is higher when the steering angle approaches to zero.)

The model also trims the images from the top (taking out the landscape) and the bottom (taking out the car's hood).

Since most of the augmented data set are generated with random alterations, every epoch generate a slightly different data set. This should also avoid over-fitting.

The validation set consists of the 20% of the center camera images *without any postprocessing*. My assumption is that the validation loss would be close to the true loss on the test set.

Finally, I set the learning rate at 0.0001 which is the recommend number for the Adam optimizer. After many attempts I found that 7 (seven) epochs were enough for the model to drive the first track successfully.

### Recording.

The results were capture in a video:

https://youtu.be/

[![Click to view!](https://img.youtube.com/vi/2wtDGQjOuwY/0.jpg)](https://www.youtube.com/watch?v=2wtDGQjOuwY)
