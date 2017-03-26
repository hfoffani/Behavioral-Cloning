import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


print('reading data...')

ISSTRAIGHT=0.05
KEEPSTRAIGHT=0.1
KEEPLATERAL=0.2
OFFSETCAMS=0.30

LEARNINGRATE=0.001
EPOCHS=7
VALIDATIONSPLIT=0.2


class monoid:

    def __init__(self, inp, func=lambda x: x):
        self.inp = inp
        self.func = func

    def __or__(self, func):
        return monoid(self, func)

    def __call__(self, inp):
        return self.func(iter(inp))

    def __iter__(self):
        return self.func(iter(self.inp))

def readcsv(fname):
    def func(inp):
        with open(fname) as csvfile:
            reader = csv.reader(csvfile)
            header = True
            for line in reader:
                if header:
                    header = False
                    continue
                yield line
    return monoid(tuple(), func)

def ufilter(f):
    def func(inp):
        return filter(f, inp)
    return monoid(tuple(), func)

def flip():
    def func(inp):
        for l in inp:
            img = l[0]
            steer = l[1]
            yield img, steer
            imageFlipped = cv2.flip(img, 1)
            yield imageFlipped, -steer
    return monoid(tuple(), func)

def readimgs():
    def func(inp):
        for line in inp:
            steer = float(line[3])
            file_center_cam = 'data/' + line[0].strip()
            file_left_cam  = 'data/' + line[1].strip()
            file_right_cam = 'data/' + line[2].strip()
            # center
            img = cv2.imread(file_center_cam)
            assert img is not None
            c_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield c_image, steer
            # left
            img = cv2.imread(file_left_cam)
            assert img is not None
            l_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield l_image, steer + OFFSETCAMS
            # right
            img = cv2.imread(file_right_cam)
            assert img is not None
    return monoid(tuple(), func)

def rem_straight():
    def func(inp):
        for im, steer in inp:
            if abs(steer) < ISSTRAIGHT and np.random.rand() > KEEPSTRAIGHT:
                continue
            yield im, steer
    return monoid(tuple(), func)

def rem_correction():
    def func(inp):
        for im, steer in inp:
            if abs((abs(steer) - OFFSETCAMS)) < ISSTRAIGHT and np.random.rand() > KEEPLATERAL:
                continue
            yield im, steer
    return monoid(tuple(), func)

def write_angles(fname):
    def func(inp):
        with open(fname, 'w') as angfile:
            angfile.write("steer\n")
            for im, steer in inp:
                angfile.write("%f\n" % steer)
                yield im, steer
    return monoid(tuple(), func)

def to_numpy(data):
    images = []
    angles = []
    for im, steer in data :
        images.append(im)
        angles.append(steer)
    assert len(images) == len(angles)
    return np.array(images), np.array(angles)


inputdata = readcsv('data/driving_log.csv') \
            | readimgs() \
            | rem_straight() \
            | rem_correction() \
            | flip() \
            | write_angles('models/angles.csv')


X_train, y_train = to_numpy(inputdata)
print('...all pre processed. # observations:', len(y_train))
print()
# exit()


def resize4nvidia(img):
    import tensorflow as tf
    return tf.image.resize_images(img, [66, 200])


model = Sequential()

# model.add(Cropping2D . couldn't make it work
model.add(Lambda(lambda x: x[:, :, 60:-30, 0:], input_shape=(160,320,3)))
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

model.add(Dense(20, activation="elu"))
model.add(Dropout(0.5))

model.add(Dense(1))

# model.summary()
# exit()

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
