import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, \
    Flatten, Input, Lambda, Cropping2D, Convolution2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import typing
from fix_path import update_df
import random
from image_preprocess import preprossing, IMAGE_HEIGHT, IMAGE_WIDTH

flags = tf.app.flags
FLAGS = flags.FLAGS

# Set parameters here
steering_index=3
parent_data_folder = './data/'
img_sub_foler = 'IMG/'
ch, row, col = 3, 160, 320

# A list of tuples of (training_folder_name,weight_of_the_datset,exclude_zero_steering)

train_dataset_folder = [("track1_new_1/", 2, False),
                        ("track1_rec_1/", 2, True),
                        ("track1_rec_2", 2, True),
                        ("track1_rec_3", 2, True),
                        ("track1_rec_4", 2, False),
                        # ("track2_7",1,False),
                        # ("track2_8",1,False),
                        # ("track2_9",1,False),
                        # ("track2_10",1,False),
                        ("track2_11", 1, False),
                        ("track2_13", 1, False),
                        ("track2_rev_1", 1, False),
                        # ("track2_rec_5",1,True),
                        # ("track2_rec_6",1,True),
                        ("track2_rec_8", 1, True),
                        ("track2_rec_7", 1, True),
                        ("track2_rec_9", 1, True),
                        ("track2_rec_10", 1, True),
                        ("track2_rec_11",1,True)]

add_on_dataset_folder=[("track1_rec_4",1,False),
                       ("track2_rec_10",1,False),
                       ("track2_rec_11",1,False)]
# ("track2_curve_1",1,False)]

# train_dataset_folder=["track1_rec_3/","track1_new_1/"]
val_dataset_folder = [("track2_val_1", 1, False)]
train_side_camera = True
batch_size = 512

center_cam = {'cam_index': 0, 'steering_adjust': 0, 'slope': 1}
right_cam = {'cam_index': 2, 'steering_adjust': -0.14, "slope": 1}
left_cam = {'cam_index': 1, 'steering_adjust': 0.14, "slope": 1}


def load_multi_dataset(data_dirs_pair: list)->pd.DataFrame:

    # Change the directory of image files in the log files.
    data_dirs = list(map(lambda x: os.path.join(parent_data_folder, x[0]), data_dirs_pair))

    all_df = []
    for i, ddir in enumerate(data_dirs):
        df = update_df(ddir)
        # Eliminate low steering angle data if data_dirs_pair[i][2] is True
        if data_dirs_pair[i][2]:
            df = df.loc[np.abs(df["steering"]) > 0.01, :]
        for k in range(data_dirs_pair[i][1]):
            all_df.append(df)

    all_df = pd.concat(all_df)

    return all_df


def filter_zero_steering(df:pd.DataFrame, keep_size=0.01, verbose=False)->pd.DataFrame:
    """
    Reduce the number of entries in the data that steering==0.0

    :param df: The pandas dataframe loaded from driving_log.csv
    :param keep_size: The ratio of df['steering']==0.0 entries to be kept
    :param verbose: Set True to display the messages
    :return:
    """
    if verbose:
        print("Filtering entries with steering value ==0.0")

    steering_values = df.iloc[:, steering_index].values

    # Separate the data frame into two groups: steering==0.0 and steering !=0.0
    idx = np.abs(steering_values) == 0.0
    non_zero_idx = np.abs(steering_values) != 0.0

    zero_len = np.sum(idx)

    sel_idx = np.random.choice(np.arange(0, df.shape[0])[idx], int(zero_len * keep_size))

    assert np.all(df.iloc[sel_idx, steering_index] == 0.0)
    assert np.all(df.iloc[non_zero_idx, steering_index] != 0.0)

    ndf = pd.concat([df.iloc[sel_idx, :], df.iloc[non_zero_idx, :]], axis=0)

    return ndf


def resample_df(df:pd.DataFrame, bins=100)->pd.DataFrame:
    """
    Resample the driving data from the histogram of steering values.
    It divides the range of steering values into a certain number of beams,
    then select the medium counts of sample from each bin.

    :param df: The pandas dataframe that has same forms of driving_log.csv
    :param bins: The number of bins
    :return: The new filtered dataframe
    """
    steering_val = df["steering"].values
    counts, bin_bound = np.histogram(steering_val, bins=bins)
    rand_size = np.floor(np.median(counts)).astype('int')

    final_idx = []

    # Go through each bin and randomly select data from each bin
    for i in range(bin_bound.shape[0] - 1):
        idx = np.logical_and(steering_val >= bin_bound[i], steering_val < bin_bound[i + 1])
        num_idx = np.flatnonzero(idx)
        sel_idx = np.random.choice(num_idx, rand_size)
        assert np.all(steering_val[sel_idx] >= bin_bound[i])
        assert np.all(steering_val[sel_idx] < bin_bound[i + 1])
        final_idx.extend(sel_idx)
    final_idx = np.array(final_idx)
    return df.iloc[final_idx, :]


def augment_brightness_camera_images(image):
    """
    Augmentation of brightness.
    This code is from
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.gv8islemq

    :param image:
    :return:
    """

    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (col, row))

    return image_tr, steer_ang


def generator(input_samples, batch_size=32,
              shuffle_samples=True, side_cam=False,
              filter=False, sample_num_bound=None,zero_size=0.01,alternating=True):

    start_filter_option=filter
    while True:  # Loop forever so the generator never terminates
        print("filter:%s"%filter)
        if alternating==True and filter==False:
            samples=filter_zero_steering(input_samples,keep_size=zero_size*10)
            print("using more zero data")
        else:
            samples=filter_zero_steering(input_samples,keep_size=zero_size)
            print("use reg. zero data")

        if filter == True:
            samples = resample_df(samples)
            sel_idx = np.random.choice(samples.shape[0], sample_num_bound)
            samples = samples.iloc[sel_idx, :]
        else:
            #samples = input_samples
            #assert sample_num_bound>samples.shape[0]
            sel_idx = np.random.choice(samples.shape[0], sample_num_bound)
            samples = samples.iloc[sel_idx, :]

        if alternating==False:
            filter=start_filter_option
        else:
            filter=(not filter)

        if shuffle_samples:
            samples = shuffle(samples)
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            if side_cam == True:
                cam = random.choice([center_cam, right_cam, left_cam])
            else:
                cam = center_cam

            batch_samples = samples.iloc[offset:min(offset + batch_size, num_samples), cam['cam_index']]

            images = []
            steer_adj = []
            for batch_sample in batch_samples:
                cam_image = cv2.imread(batch_sample)
                # augment new images
                cam_image = augment_brightness_camera_images(cam_image)
                cam_image, adj = trans_image(cam_image, 100)
                if np.random.rand()>0.5:
                    cam_image=np.flip(cam_image,axis=2)

                # Proprocess the image for training
                cam_image = preprossing(cam_image)

                images.append(cam_image)
                steer_adj.append(adj)

            X_train = np.array(images)
            y_train = samples.iloc[offset:min(offset + batch_size, num_samples), 3] \
                      * cam["slope"] + cam['steering_adjust'] + np.array(steer_adj)

            yield X_train, y_train



def main(_):

    train_samples = load_multi_dataset(train_dataset_folder)
    #train_samples=load_multi_dataset(add_on_dataset_folder)
    validation_samples = load_multi_dataset(val_dataset_folder)

    # train_samples=filter_dataset(train_samples)
    dummy_train_samples = filter_zero_steering(train_samples)
    dummy_train_samples = resample_df(dummy_train_samples)
    var_sample_num = dummy_train_samples.shape[0]

    d2_train_samples=filter_zero_steering(train_samples,keep_size=0.1)


    train_generator_filter = generator(train_samples,
                                batch_size=batch_size,
                                side_cam=train_side_camera, filter=True, sample_num_bound=var_sample_num)

    validation_generator = generator(validation_samples,
                                     batch_size=batch_size, side_cam=False, filter=False,
                                     sample_num_bound=validation_samples.shape[0],alternating=False)

    nvidia_model = Sequential()

    nvidia_model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

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

    # adam=Adam(lr=0.0001)
    nvidia_model.compile(optimizer='adam', loss='mse')
    #nvidia_model.load_weights('nvidia_model_weights_r_v13.h5')

    checkpoint = ModelCheckpoint(filepath='./_model_checkpoints/model-{epoch:02d}.h5')
    callback_list = [checkpoint]

    hist = nvidia_model.fit_generator(train_generator_filter,
                                          var_sample_num,
                                          nb_epoch=10,
                                          validation_data=validation_generator,
                                          nb_val_samples=validation_samples.shape[0],
                                          callbacks=callback_list)

    # nvidia_model.evaluate_generator(validation_generator,validation_samples.shape[0])

    with open('model_hist_r_v14.p', 'wb') as fp:
        pickle.dump(hist.history, fp)

    nvidia_model.save("model_r_v14.h5")
    nvidia_model.save_weights('nvidia_model_weights_r_v14.h5')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
