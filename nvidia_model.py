import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, \
    Flatten, Input, Lambda, Cropping2D, Convolution2D
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import typing
from fix_path import update_df

flags = tf.app.flags
FLAGS = flags.FLAGS

# Set parameters here
parent_data_folder = './data/'
img_sub_foler = 'IMG/'
ch, row, col = 3, 160, 320
ch, p_row, p_col = 3, 160, 320
train_dataset_folder = ["official_baseline/"]
batch_size = 128


def load_multi_dataset(data_dirs: list):
    data_dirs = map(lambda x: os.path.join(parent_data_folder, x), data_dirs)

    all_df = []
    for ddir in data_dirs:
        df = update_df(ddir)
        all_df.append(df)

    all_df = pd.concat(all_df)

    return all_df

def filter_dataset(df):

    idx=np.abs(df.iloc[:,3].values)>0.01

    ndf=df.iloc[idx,:]

    ndf=pd.concat([df,ndf],axis=0)

    return ndf



def load_sample_df(df: pd.DataFrame, test_size=0.2):
    train, val = train_test_split(df, test_size=test_size)

    return train, val


def generator(samples, batch_size=32, shuffle_samples=True):
    num_samples = len(samples)

    while True:  # Loop forever so the generator never terminates
        if shuffle_samples:
            print("reshuffled")
            samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:min(offset + batch_size, num_samples), 0]

            images = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample)
                images.append(center_image)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = samples.iloc[offset:min(offset + batch_size, num_samples), 3]

            #filter out index that steering>0
            #idx=np.abs(y_train.values)>0.01

            nX_train=np.flip(X_train,axis=3)

            cX_train=np.concatenate((X_train,nX_train),axis=0)
            cy_train=np.concatenate((y_train,y_train),axis=0)

            yield cX_train, cy_train


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val



def main(_):
    # load bottleneck data

    df = load_multi_dataset(train_dataset_folder)

    train_samples, validation_samples = load_sample_df(df, test_size=0.2)

    train_samples=filter_dataset(train_samples)

    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    nvidia_model = Sequential()

    nvidia_model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))

    nvidia_model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    nvidia_model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    nvidia_model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    nvidia_model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    nvidia_model.add(Convolution2D(64, 3, 3, activation='relu'))
    nvidia_model.add(Convolution2D(64, 3, 3, activation='relu'))

    nvidia_model.add(Flatten())
    nvidia_model.add(Dense(100))
    nvidia_model.add(Dropout(0.5))
    nvidia_model.add(Dense(50))
    nvidia_model.add(Dropout(0.5))
    nvidia_model.add(Dense(10))
    nvidia_model.add(Dense(1))

    nvidia_model.compile(optimizer='adam', loss='mse')
    # nvidia_model.load_weights('nvidia_model_weights_v2.h5')
    hist = nvidia_model.fit_generator(train_generator, train_samples.shape[0]*2, nb_epoch=10,
                                      validation_data=validation_generator,nb_val_samples=validation_samples.shape[0])
    # nvidia_model.evaluate_generator(validation_generator,validation_samples.shape[0])

    nvidia_model.save("model_v4.h5")
    nvidia_model.save_weights('nvidia_model_weights_v4.h5')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
