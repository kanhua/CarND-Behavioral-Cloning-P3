import pickle
import numpy as np
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


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


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    #assert class_n==class_n2

    sq=Sequential()
    sq.add(Flatten(input_shape=(X_train[0].shape)))

    sq.add(Dense(output_dim=200))
    sq.add(Dense(output_dim=100))
    sq.add(Dense(output_dim=50))
    sq.add(Dense(output_dim=10))
    sq.add(Dense(output_dim=1))

    print(sq.layers[0].output_shape)
    print(sq.layers[1].output_shape)
    sq.compile(optimizer='adam', loss='mse')

    sq.fit(X_train, y_train, batch_size=128, nb_epoch=20,validation_data=(X_val,y_val))

    sq.save("test_trans_model.h5")

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
