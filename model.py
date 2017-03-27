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


np.random.seed(5)

print('reading data...')

ISSTRAIGHT=0.05
KEEPSTRAIGHT=0.1
KEEPLATERAL=0.2
OFFSETCAMS=0.20

LEARNINGRATE=0.001
EPOCHS=7
VALIDATIONSPLIT=0.2



class Pipe:
    """ From https://github.com/JulienPalard/Pipe
    """

    def __init__(self, func=lambda x: x):
        self.func = func

    def __ror__(self, other):
        return self.func(other)

    def __call__(self, *args, **kwargs):
        return Pipe(lambda x: self.func(x, *args, **kwargs))


def readcsv(fname):
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for line in reader:
            if header:
                header = False
                continue
            yield line

@Pipe
def flip_images_horizontally(iterable):
    for l in iterable:
        img = l[0]
        steer = l[1]
        yield img, steer
        imageFlipped = cv2.flip(img, 1)
        yield imageFlipped, -steer

@Pipe
def read_images_and_steer(iterable):
    for line in iterable:
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

@Pipe
def remove_straight(iterable):
    for im, steer in iterable:
        if abs(steer) < ISSTRAIGHT and np.random.rand() > KEEPSTRAIGHT:
            continue
        yield im, steer

@Pipe
def remove_left_right_from_straight(iterable):
    for im, steer in iterable:
        if abs((abs(steer) - OFFSETCAMS)) < ISSTRAIGHT and np.random.rand() > KEEPLATERAL:
            continue
        yield im, steer

@Pipe
def write_angles_to_file(iterable, fname):
    with open(fname, 'w') as angfile:
        angfile.write("steer\n")
        for im, steer in iterable:
            angfile.write("%f\n" % steer)
            yield im, steer


#
# INPUT DATA PIPELINE
#
inputdata = readcsv('data/driving_log.csv') \
            | read_images_and_steer() \
            | remove_straight() \
            | remove_left_right_from_straight() \
            | flip_images_horizontally() \
            | write_angles_to_file('models/angles.csv')


def to_numpy(data):
    images = []
    angles = []
    for im, steer in data :
        images.append(im)
        angles.append(steer)
    assert len(images) == len(angles)
    return np.array(images), np.array(angles)


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
