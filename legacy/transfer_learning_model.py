import pickle
import numpy as np
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Flatten,Input,Lambda
from keras.applications.vgg16 import VGG16
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input
import cv2
import keras.backend as K
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS


sample_folder = './lindata/'

ch, row, col = 3, 160, 320  # Trimmed image format

p_ch, p_row, p_col = 3, 160, 320
img_placeholder = tf.placeholder("uint8", (None, 160, 320, 3))

resize_op = tf.image.resize_images(img_placeholder, (p_row, p_col), method=0)
#single_img_placeholder=tf.placeholder("uint8",(160,320,3))
#resize_op = tf.image.resize_image_with_crop_or_pad(single_img_placeholder, p_row, p_col)


def load_samples_df(test_size=0.2):
    sample_df = pd.DataFrame.from_csv(os.path.join(sample_folder, 'driving_log.csv'), index_col=None)

    train, val = train_test_split(sample_df, test_size=test_size)

    return train, val


def generator(samples, batch_size=32):
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

            y_train=samples.iloc[offset:min(offset + batch_size,num_samples), 3]

            yield X_train,y_train



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

def y_to_label(y_data):

    new_y=np.zeros_like((y_data))
    new_y[y_data==0]=int(2)
    new_y[y_data>0]=int(0)
    new_y[y_data<0]=int(1)

    return new_y

def resize_image(input_image):

    #input_x = sess.run(resize_op, feed_dict={img_placeholder: input_image})
    resized_img=tf.image.resize_images(input_image, (p_row, p_col), method=0)

    return resized_img



def main(_):
    # load bottleneck data


    train_samples, validation_samples = load_samples_df(test_size=0.2)

    batch_size=128
    small_batch_size=12



    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    preprocess=Sequential()
    #preprocess.add(Input(shape=(p_row, p_col, ch)))
    #preprocess.add(Lambda(resize_image,input_shape=(p_row, p_col, ch)))
    preprocess.add(Lambda(lambda x:x/255.0-0.5,input_shape=(row, col, ch)))

    pre_trained_model = VGG16(input_tensor=preprocess.output, include_top=False)
    #pre_trained_model = VGG16(input_tensor=Input(shape=(row, col, ch)), include_top=False)
    x = pre_trained_model.output
    x=Flatten()(x)
    x=Dense(100)(x)
    x=Dense(100)(x)
    x=Dense(1)(x)
    #model = Model(pre_trained_model.input, x)
    model = Model(preprocess.input, x)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    for layer in preprocess.layers:
        layer.trainable = False

    model.compile(optimizer='adam',loss='mse')
    hist=model.fit_generator(train_generator,train_samples.shape[0],nb_epoch=10)
    #hist=model.fit_generator(train_generator,small_batch_size*1,nb_epoch=1)
    #pickle.dump(hist,open('hist.p','wb'))
    model.save("test_model.h5")



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
