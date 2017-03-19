import csv
import cv2
import numpy as np

print('reading data...')

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = True
    for line in reader:
        if header:
            header = False
            continue
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = 'data/' + source_path
    image = cv2.imread(filename)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print('...all pre processed')


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import Adam

def resize4nvidia(img):
    import tensorflow as tf
    return tf.image.resize_images(img, [66, 200])

model = Sequential()
# model.add(Cropping2D . couldn't make it work
model.add(Lambda(lambda x: x[:, :, 60:, :], input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Lambda(resize4nvidia))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu")) 
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())

model.add(Dense(100, activation="elu"))
model.add(Dropout(0.5))

model.add(Dense(50, activation="elu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation="elu"))

model.add(Dense(1))


model.summary()
model.compile(loss='mse',
            optimizer=Adam(lr=0.0001))
model.fit(X_train, y_train,
            validation_split=0.2,
            shuffle=True,
            nb_epoch=5)

model.save('model.h5')

exit()
