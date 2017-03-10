import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,\
    Flatten,Input,Lambda,Cropping2D,Convolution2D
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import cv2
import typing
from fix_path import update_df

flags = tf.app.flags
FLAGS = flags.FLAGS


parent_data_folder='./data/'
img_sub_foler='IMG/'

ch, row, col = 3, 160, 320  # Trimmed image format

p_ch, p_row, p_col = 3, 160, 320
img_placeholder = tf.placeholder("uint8", (None, 160, 320, 3))

resize_op = tf.image.resize_images(img_placeholder, (p_row, p_col), method=0)
#single_img_placeholder=tf.placeholder("uint8",(160,320,3))
#resize_op = tf.image.resize_image_with_crop_or_pad(single_img_placeholder, p_row, p_col)


def load_multi_dataset(data_dirs:list):


    data_dirs=map(lambda x: os.path.join(parent_data_folder,x),data_dirs)

    all_df=[]
    for dir in data_dirs:
        df=update_df(dir)
        all_df.append(df)

    all_df=pd.concat(all_df)

    return all_df

def load_sample_df(df:pd.DataFrame,test_size=0.2):

    train, val = train_test_split(df, test_size=test_size)

    return train,val


def generator(samples, batch_size=32):
    num_samples = len(samples)
    dummy_seed=1
    while dummy_seed==1:  # Loop forever so the generator never terminates
        #dummy_seed-=1
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:min(offset + batch_size,num_samples), 0]

            images = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample)
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

    train_dataset_folder=["official_baseline/","trip1_middle/"]

    df=load_multi_dataset(train_dataset_folder)

    train_samples, validation_samples = load_sample_df(df,test_size=0.2)

    batch_size=128


    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    nvidia_model=Sequential()


    nvidia_model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(row, col, ch)))

    nvidia_model.add(Cropping2D(cropping=((70,25),(0,0))))
    nvidia_model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    nvidia_model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    nvidia_model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))

    nvidia_model.add(Convolution2D(64,3,3,activation='relu'))

    nvidia_model.add(Convolution2D(64,3,3,activation='relu'))

    nvidia_model.add(Flatten())
    nvidia_model.add(Dense(100))
    nvidia_model.add(Dense(50))
    nvidia_model.add(Dense(10))
    nvidia_model.add(Dense(1))

    nvidia_model.compile(optimizer='adam',loss='mse')
    #nvidia_model.load_weights('nvidia_model_weights_v2.h5')
    hist=nvidia_model.fit_generator(train_generator,train_samples.shape[0],nb_epoch=10)
    #nvidia_model.evaluate_generator(validation_generator,validation_samples.shape[0])
    #pickle.dump(hist,open('hist.p','wb'))


    nvidia_model.save("model_test.h5")
    #nvidia_model.save_weights('nvidia_model_weights_v3.h5')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
