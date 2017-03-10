"""
Fix the path issues in the log files.
This is for the preparation of running batch training

"""

import pandas as pd
import os


def update_df(log_file_dir):
    """
    Update the file directory of each image file in driving_log.csv.
    This function returns a new dataframe that has the same form as driving_log.csv.
    The file name of each image file becomes: log_file_dir/IMG/XXXXXX.JPG
    :param log_file_dir: the new file directory.
    :return: the new dataframe
    """

    def change_dir(x):
        file_dir, fname = os.path.split(x)
        return os.path.join(log_file_dir, 'IMG/', fname)

    df = pd.DataFrame.from_csv(os.path.join(log_file_dir, 'driving_log.csv'), header=0, index_col=None)

    ndf = df[['center', 'left', 'right']]

    ndf = ndf.applymap(change_dir)

    df[['center', 'left', 'right']] = ndf

    return df


if __name__ == "__main__":
    file_dir = './data/official_baseline/'

    df = update_df(file_dir)

    df.to_csv('new_log_file.csv', index=None)
