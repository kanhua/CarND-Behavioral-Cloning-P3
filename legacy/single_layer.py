"""
A single layer model, essentially a regression model, to predict the steering images.
Set the variable 'AUGMENT' to determine if you want to run image augmentation.
If AUGMENT is set to be True, the script will flip the images horizontally in each batch and add it to the training data.
"""

import os
import cv2
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
import numpy as np

AUGMENT=True

sample_folder = './sample behavioral cloning data/'

ch, row, col = 3, 160, 320


def load_samples_df(test_size):
    sample_df = pd.DataFrame.from_csv(os.path.join(sample_folder, 'driving_log.csv'), index_col=None)

    train, val = train_test_split(sample_df, test_size=test_size)

    return train, val


def generator(samples, batch_size=32, augment=False):
    num_samples = len(samples)
    dummy_seed = 1
    while dummy_seed == 1:  # Loop forever so the generator never terminates
        # dummy_seed-=1
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:min(offset + batch_size, num_samples), 0]

            images = []
            for batch_sample in batch_samples:
                name = os.path.join(sample_folder, batch_sample)
                center_image = cv2.imread(name)
                # cv2.imwrite('test.jpg',center_image)
                images.append(center_image)

            # trim image to only see section with road
            X_train = np.array(images).astype('float64')
            y_train = samples.iloc[offset:min(offset + batch_size, num_samples), 3]
            # X_train = preprocess_input(X_train)

            if augment == True:
                inv_X_train = np.flip(X_train, axis=1)
                inv_y_train = -y_train
                X_train = np.concatenate((X_train, inv_X_train), axis=0)
                y_train = np.concatenate((y_train, inv_y_train), axis=0)

            # plt.imshow(X_train[0])
            # plt.savefig("test2.jpg")

            yield X_train, y_train


t, v = load_samples_df(test_size=0.2)
train_data_g = generator(t,augment=AUGMENT)
val_data_g = generator(v,augment=AUGMENT)
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(row, col, ch)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit_generator(generator=train_data_g, samples_per_epoch=t.shape[0] * (1+int(AUGMENT)), nb_epoch=2,
                           validation_data=val_data_g, nb_val_samples=v.shape[0] * (1+int(AUGMENT)))

model.save("test_model.h5")
