import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


print('reading data...')

KEEPSTRAIGHT=0.1
OFFSETCAMS=0.2

LEARNINGRATE=0.0005
EPOCHS=7
VALIDATIONSPLIT=0.1

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = True
    for line in reader:
        if header:
            header = False
            continue
        lines.append(line)

def get_cams(lines):
    for line in lines:
        steer = float(line[3])
        if abs(round(steer,2)) == 0.0 and np.random.rand() > KEEPSTRAIGHT:
            continue
        file_center_cam = 'data/' + line[0].strip()
        file_left_cam  = 'data/' + line[1].strip()
        file_right_cam = 'data/' + line[2].strip()
        c_image = cv2.imread(file_center_cam)
        yield c_image, steer 
        l_image = cv2.imread(file_left_cam)
        yield l_image, steer + OFFSETCAMS
        r_image = cv2.imread(file_right_cam)
        yield r_image, steer - OFFSETCAMS


def add_rows(obs):
    for im, steer in obs:
        yield im, steer
        imageFlipped = cv2.flip(im, 1)
        yield imageFlipped, -steer
        

def filter_rows(obs):
    for im, steer in obs:
        yield im, steer
        

images = []
angles = []

with open('models/angles.csv', 'w') as angfile:
    angfile.write("steer\n")
    for im, steer in filter_rows( add_rows( get_cams( lines ))):
        images.append(im)
        angles.append(steer)
        angfile.write("%f\n" % steer)

assert len(images) == len(angles)

X_train = np.array(images)
y_train = np.array(angles)
print('...all pre processed. # observations:', len(angles))
# exit()


def get_model():

    def resize4nvidia(img):
        import tensorflow as tf
        return tf.image.resize_images(img, [66, 200])

    model = Sequential()

    # model.add(Cropping2D . couldn't make it work
    model.add(Lambda(lambda x: x[:, :, 60:, :], input_shape=(160,320,3)))
    model.add(Lambda(resize4nvidia))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2),
                init="he_normal", activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2),
                init="he_normal", activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2),
                init="he_normal", activation="relu"))
    model.add(Convolution2D(64, 3, 3,
                init="he_normal", activation="relu"))
    model.add(Convolution2D(64, 3, 3,
                init="he_normal", activation="relu"))

    model.add(Flatten())

    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation="elu"))

    model.add(Dense(1))

    # model.summary()
    return model


model = get_model()
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
