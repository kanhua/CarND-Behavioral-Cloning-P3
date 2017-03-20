import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, \
    Flatten, Input, Lambda, Cropping2D, Convolution2D
from keras.callbacks import ModelCheckpoint
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import typing
from fix_path import update_df
import random
import keras.backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS

# Set parameters here
parent_data_folder = './data/'
img_sub_foler = 'IMG/'
ch, row, col = 3, 160, 320
ch, p_row, p_col = 3, 80, 160
train_dataset_folder = [("track1_new_1/",1),
                        ("track1_rec_1/",2),
                        ("track1_rec_2",2),
                        ("track1_rec_3",2),
                        ("track2_7",1),
                        ("track2_8",1),
                        ("track2_9",1),
                        ("track2_rec_5",2),
                        ("track2_rec_6",2),
                        ("track2_rec_8",2),
                        ("track2_rec_7",2)]
#train_dataset_folder=["track1_rec_3/","track1_new_1/"]
val_dataset_folder=[("official_baseline",1)]
train_side_camera=True
batch_size = 128


def load_multi_dataset(data_dirs_pair:list):
    data_dirs = list(map(lambda x: os.path.join(parent_data_folder, x[0]), data_dirs_pair))

    all_df = []
    for i,ddir in enumerate(data_dirs):
        df = update_df(ddir)
        for k in range(data_dirs_pair[i][1]):
            all_df.append(df)

    all_df = pd.concat(all_df)

    return all_df

def filter_dataset(df,portion=100,verbose=False):

    if verbose:
        print("samples refiltered")

    steering_values=df.iloc[:,3].values

    idx=np.abs(steering_values)==0.0
    non_zero_idx=np.abs(steering_values)!=0.0

    zero_len=np.sum(idx)

    sel_idx=np.random.choice(np.arange(0,df.shape[0])[idx],int(zero_len/portion))

    assert np.all(df.iloc[sel_idx,3]==0.0)
    assert np.all(df.iloc[non_zero_idx,3]!=0.0)
    ndf=pd.concat([df.iloc[sel_idx,:],df.iloc[non_zero_idx,:]],axis=0)

    return ndf

def filter_dataset_2nd_pass(df,portion=2,verbose=False):

    if verbose:
        print("samples refiltered")

    steering_values=df.iloc[:,3].values

    idx=np.abs(steering_values)<0.2
    non_zero_idx=np.abs(steering_values)>0.2

    zero_len=np.sum(idx)

    sel_idx=np.random.choice(np.arange(0,df.shape[0])[idx],int(zero_len/portion))

    assert np.all(np.abs(df.iloc[sel_idx,3])<0.2)
    assert np.all(np.abs(df.iloc[non_zero_idx,3])>0.2)
    ndf=pd.concat([df.iloc[sel_idx,:],df.iloc[non_zero_idx,:]],axis=0)

    return ndf


def augment_brightness_camera_images(image):
    """
    Augmentation of brightness.
    This code is from
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.gv8islemq

    :param image:
    :return:
    """

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1



def load_sample_df(df: pd.DataFrame, test_size=0.2):
    train, val = train_test_split(df, test_size=test_size)

    return train, val


center_cam={'cam_index':0,'steering_adjust':0}
right_cam={'cam_index':2,'steering_adjust':-0.16}
left_cam={'cam_index':1,'steering_adjust':0.16}


def generator(input_samples, batch_size=32, shuffle_samples=True,side_cam=False,filter=False):

    while True:  # Loop forever so the generator never terminates
        if filter==True:
            samples = filter_dataset(input_samples)
            samples = filter_dataset_2nd_pass(samples)
        else:
            samples = input_samples

        if shuffle_samples:
            samples = shuffle(samples)
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            if side_cam==True:
                cam=random.choice([center_cam,right_cam,left_cam])
            else:
                cam=center_cam

            batch_samples = samples.iloc[offset:min(offset + batch_size, num_samples), cam['cam_index']]

            images = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample)

                center_image=cv2.resize(center_image,dsize=(0,0),fx=0.5,fy=0.5)
                #augment new images
                center_image=augment_brightness_camera_images(center_image)

                images.append(center_image)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = samples.iloc[offset:min(offset + batch_size, num_samples), 3]+cam['steering_adjust']

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

    train_samples = load_multi_dataset(train_dataset_folder)

    validation_samples=load_multi_dataset(val_dataset_folder)

    #train_samples=filter_dataset(train_samples)
    dummy_train_samples=filter_dataset(train_samples)
    dummy_train_samples=filter_dataset_2nd_pass(dummy_train_samples)
    var_sample_num=dummy_train_samples.shape[0]

    train_generator = generator(train_samples,
                                batch_size=batch_size,side_cam=train_side_camera,filter=True)
    validation_generator = generator(validation_samples,
                                     batch_size=batch_size,side_cam=False,filter=False)


    nvidia_model = Sequential()


    nvidia_model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(p_row, p_col, ch)))

    #nvidia_model.add(Cropping2D(cropping=((70, 24), (0, 0))))
    nvidia_model.add(Cropping2D(cropping=((0, 8), (0, 0))))

    #nvidia_model.add(Lambda(lambda image: K.resize_images(image, (160-94)/2,160,'tf')))
    nvidia_model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    nvidia_model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    nvidia_model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    nvidia_model.add(Convolution2D(64, 3, 3, activation='relu'))
    nvidia_model.add(Convolution2D(64, 3, 3, activation='relu'))

    nvidia_model.add(Flatten())
    nvidia_model.add(Dense(100))
    nvidia_model.add(Dense(50))
    nvidia_model.add(Dense(10))
    nvidia_model.add(Dense(1))

    nvidia_model.compile(optimizer='adam', loss='mse')
    #nvidia_model.load_weights('nvidia_model_weights_r_v4.h5')

    checkpoint = ModelCheckpoint(filepath='./_model_checkpoints/model-{epoch:02d}.h5')
    callback_list = [checkpoint]

    hist = nvidia_model.fit_generator(train_generator,
                                      var_sample_num*2,
                                      nb_epoch=15,
                                      validation_data=validation_generator,
                                      nb_val_samples=validation_samples.shape[0]*2,
                                      callbacks=callback_list)

    # nvidia_model.evaluate_generator(validation_generator,validation_samples.shape[0])

    #with open('model_hist.p','wb') as fp:
    #    pickle.dump(hist['loss'],fp)

    nvidia_model.save("model_r_v5.h5")
    nvidia_model.save_weights('nvidia_model_weights_r_v5.h5')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

