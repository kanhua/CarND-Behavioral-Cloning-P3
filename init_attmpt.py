import os
import csv
import cv2
import numpy as np
from keras.layers import Input, AveragePooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import pandas as pd

sample_folder = './sample behavioral cloning data/'

ch, row, col = 3, 160, 320  # Trimmed image format

p_ch, p_row, p_col = 3, 80, 160
img_placeholder = tf.placeholder("uint8", (None, 160, 320, 3))

resize_op = tf.image.resize_images(img_placeholder, (p_row, p_col), method=0)
#single_img_placeholder=tf.placeholder("uint8",(160,320,3))
#resize_op = tf.image.resize_image_with_crop_or_pad(single_img_placeholder, p_row, p_col)


def load_samples():
    samples = []

    with open(os.path.join(sample_folder, './driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


def load_samples_df(test_size=0.2):
    sample_df = pd.DataFrame.from_csv(os.path.join(sample_folder, 'driving_log.csv'), index_col=None)

    train, val = train_test_split(sample_df, test_size=test_size)

    return train, val


def generator(sess, samples, batch_size=32):
    num_samples = len(samples)
    dummy_seed=1
    while dummy_seed==1:  # Loop forever so the generator never terminates
        #dummy_seed-=1
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:min(offset + batch_size,num_samples), 0]

            images = []
            for batch_sample in batch_samples:
                name = os.path.join(sample_folder, batch_sample)
                center_image = cv2.imread(name)
                #center_image=sess.run(resize_op,feed_dict={single_img_placeholder:center_image})
                #cv2.imwrite('test.jpg',center_image)
                images.append(center_image)

            # trim image to only see section with road
            X_train = np.array(images)
            X_train=sess.run(resize_op,feed_dict={img_placeholder:X_train})
            y_train=samples.iloc[offset:min(offset + batch_size,num_samples), 3]
            X_train = preprocess_input(X_train.astype('float64'))

            #plt.imshow(X_train[0])
            #plt.savefig("test2.jpg")

            yield X_train,y_train


def create_model(trained_model, row, col, ch):
    input_tensor = Input(shape=(row, col, ch))
    if trained_model == 'vgg':
        model = VGG16(input_tensor=input_tensor, include_top=False)
        x = model.output
        #x = AveragePooling2D((7, 7))(x)
        model = Model(model.input, x)
    elif trained_model == 'inception':
        model = InceptionV3(input_tensor=input_tensor, include_top=False)
        x = model.output
        x = AveragePooling2D((8, 8), strides=(8, 8))(x)
        model = Model(model.input, x)
    else:
        model = ResNet50(input_tensor=input_tensor, include_top=False)
    return model


def main():
    train_output_file = 'test_train_n.p'
    validation_output_file = 'test_val_n.p'

    train_samples, validation_samples = load_samples_df(test_size=0.2)

    with tf.Session() as sess:
        K.set_session(sess)
        K.set_learning_phase(1)

        model = create_model('vgg', ch=p_ch, row=p_row, col=p_col)

        batch_size=32
        # compile and train the model using the generator function
        train_generator = generator(sess, train_samples, batch_size=batch_size)
        validation_generator = generator(sess, validation_samples, batch_size=batch_size)

        import math
        print('Bottleneck training')
        bottleneck_features_train = model.predict_generator(train_generator, train_samples.shape[0])
        #bottleneck_features_train = model.predict_generator(train_generator, var_train_sample)
        data = {'features': bottleneck_features_train, 'labels': train_samples['steering'].values}
        pickle.dump(data, open(train_output_file, 'wb'))

        print('Bottleneck validation')
        var_val_sample=math.floor(len(validation_samples)/batch_size)*batch_size
        #var_val_sample=batch_size*2
        bottleneck_features_validation = model.predict_generator(validation_generator,validation_samples.shape[0])

        data = {'features': bottleneck_features_validation,
                'labels': validation_samples['steering'].values}
        pickle.dump(data, open(validation_output_file, 'wb'))


if __name__ == "__main__":
    main()

"""
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=(ch, row, col),
                 output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
len(train_samples), validation_data=validation_generator, /
nb_val_samples=len(validation_samples), nb_epoch=3)
"""
