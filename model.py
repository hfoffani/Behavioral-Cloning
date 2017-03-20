import csv
import cv2
import numpy as np

print('reading data...')

ADDFLIPPED=0.3
ISSTRAIGHT=0.1
KEEPSTRAIGHT=0.9
OFFSETCAMS=0.2

LEARNINGRATE=0.0001
EPOCHS=5
VALIDATIONSPLIT=0.2

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
    measurement = float(line[3])
    if abs(measurement) < ISSTRAIGHT and np.random.rand() > KEEPSTRAIGHT:
        continue
    measurements.append(measurement)
    source_path = line[0]
    filename = 'data/' + source_path
    image = cv2.imread(filename)
    images.append(image)

imagesFlipped=[]
measurementsFlipped = []
for i in range(len(images)):
    if abs(measurements[i]) > ADDFLIPPED:
        imageFlipped = cv2.flip(images[i], 1)
        imagesFlipped.append(imageFlipped)
        measurementsFlipped.append(-measurements[i])

images.extend(imagesFlipped)
measurements.extend(measurementsFlipped)

X_train = np.array(images)
y_train = np.array(measurements)

print('...all pre processed')


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def resize4nvidia(img):
    import tensorflow as tf
    return tf.image.resize_images(img, [66, 200])

model = Sequential()

# model.add(Cropping2D . couldn't make it work
model.add(Lambda(lambda x: x[:, :, 60:, :], input_shape=(160,320,3)))
model.add(Lambda(resize4nvidia))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

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


# model.summary()
model.compile(loss='mse',
            optimizer=Adam(lr=LEARNINGRATE))
checkpoint_path="models/weights-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(checkpoint_path,
            verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
model.fit(X_train, y_train,
            validation_split=VALIDATIONSPLIT,
            shuffle=True,
            callbacks=[checkpoint],
            nb_epoch=EPOCHS)

model.save('models/model.h5')

exit()
