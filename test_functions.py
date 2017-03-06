import unittest
from init_attmpt import load_samples_df
import numpy as np
import os
import cv2


sample_folder = './sample behavioral cloning data/'

def generator(samples, batch_size=32):
    num_samples = len(samples)
    dummy_seed=1
    while dummy_seed==1:  # Loop forever so the generator never terminates
        dummy_seed-=1
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:min(offset + batch_size,num_samples), 0]

            images = []
            for batch_sample in batch_samples:
                name = os.path.join(sample_folder, batch_sample)
                center_image = cv2.imread(name)
                images.append(center_image)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train=samples.iloc[offset:min(offset + batch_size,num_samples), 3]

            yield X_train,y_train



class MyTestCase(unittest.TestCase):
    def test_generator(self):
        train_samples, validation_samples = load_samples_df(test_size=0.9)

        accum_sample_num=0
        g=generator(train_samples, batch_size=24)
        while accum_sample_num<train_samples.shape[0]:
            X,y=next(g)
            accum_sample_num+=X.shape[0]


        self.assertEqual(train_samples.shape[0], accum_sample_num)


if __name__ == '__main__':
    unittest.main()
