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

ISSTRAIGHT=0.1
KEEPSTRAIGHT=0.1
KEEPLATERAL=0.15
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
    train = []
    valid = []
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for line in reader:
            if header:
                header = False
                continue
            steer = float(line[3])
            c_cam = 'data/' + line[0].strip()
            l_cam  = 'data/' + line[1].strip()
            r_cam = 'data/' + line[2].strip()
            observ = c_cam, l_cam, r_cam, steer
            if np.random.uniform() < VALIDATIONSPLIT:
                valid.append( observ )
            else:
                train.append( observ )
    return train, valid


def img_from_filename(fname):
    img = cv2.imread(fname)
    assert img is not None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@Pipe
def read_images_and_steer(iterable, only_center_cam=False):
    for c_cam, l_cam, r_cam, steer in iterable:
        # center
        c_image = img_from_filename(c_cam)
        yield c_image, steer
        if only_center_cam:
            continue
        # left
        l_image = img_from_filename(l_cam)
        yield l_image, steer + OFFSETCAMS
        # right
        r_image = img_from_filename(r_cam)
        yield r_image, steer - OFFSETCAMS

@Pipe
def flip_images_horizontally(iterable, skip=False, include_original=True):
    for im, steer in iterable:
        if include_original:
            yield im, steer
        if skip:
            continue
        imageFlipped = cv2.flip(im, 1)
        yield imageFlipped, -steer

@Pipe
def add_translated_images(iterable,trans_range, skip=False, include_original=True):
    for im, steer in iterable:
        if include_original:
            yield im, steer
        if skip:
            continue
        tr_x = trans_range*np.random.uniform()-trans_range/2
        steer_tr = steer + tr_x/trans_range*2*.2
        # tr_y = 40*np.random.uniform()-40/2
        tr_y = 0
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        image_tr = cv2.warpAffine(im,Trans_M,(320,160))
        yield image_tr,steer_tr


@Pipe
def add_brightness_images(iterable, skip=False, include_original=True):
    for im, steer in iterable:
        if include_original:
            yield im, steer
        if skip:
            continue
        image_br = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        # image_br = np.array(image_br, dtype = np.float64)
        brightness = .5 + np.random.uniform()
        image_br[:,:,2] = int(double(image_br[:,:,2]) * brightness)
        image_br[:,:,2][image_br[:,:,2]>255] = 255
        # image_br = np.array(image_br, dtype = np.uint8)
        image_br = cv2.cvtColor(image_br, cv2.COLOR_HSV2RGB)
        yield image_br, steer

@Pipe
def remove_straight(iterable, skip=False):
    for im, steer in iterable:
        if not skip and abs(steer) < ISSTRAIGHT and np.random.rand() > KEEPSTRAIGHT:
            continue
        yield im, steer

@Pipe
def remove_left_right_from_straight(iterable, skip=False):
    for im, steer in iterable:
        if not skip and abs((abs(steer) - OFFSETCAMS)) < ISSTRAIGHT and np.random.rand() > KEEPLATERAL:
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
train_data, valid_data = readcsv('data/driving_log.csv')
validationset = valid_data \
            | read_images_and_steer(only_center_cam=True)

training = train_data \
            | read_images_and_steer() \
            | remove_straight(skip=True) \
            | remove_left_right_from_straight(skip=False) \
            | add_translated_images(100, skip=True) \
            | add_brightness_images(skip=True) \
            | flip_images_horizontally(skip=False) \
            | remove_straight() \
            | write_angles_to_file('models/angles.csv')


X_val, y_val = tuple( np.array(x) for x in zip(*validationset) )
print(len(X_val), len(y_val))
exit()

def to_numpy(data):
    images = []
    angles = []
    for im, steer in data :
        images.append(im)
        angles.append(steer)
        pass
    assert len(images) == len(angles)
    xv, yv = zip(*validationset)
    assert len(xv) == len(yv)
    return np.array(images), np.array(angles), np.array(xv), np.array(yv)


X_train, y_train, X_val, y_val = to_numpy(training)
print('...all pre processed. # observations:', len(y_train), ' validate:', len(y_val))
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
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=[checkpoint],
            nb_epoch=EPOCHS)

model.save('models/model.h5')

exit()
