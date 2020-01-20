import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os


IMG_SIZE = 64
N_CHANNELS = 1

class dataset():
    def __init__(self, name):
        self.name = name

    def load_dataset(self):
        dir = "/content/gdrive/Shared drives/2020_Bengali/dataset/"
        for i in range(4):
            path = dir + self.name+"_image_data_"+str(i)+".parquet"
            if i==0:
                df = pd.read_parquet(path)
            else:
                df = pd.concat([df, pd.read_parquet(path)])
        return df


    def resize(self, df, size = IMG_SIZE):
        resized = {}
        for i in tqdm(range(df.shape[0])):
            image = cv2.resize(df.loc[df.index[i]].values.reshape(137, 236),
                               (size, size))
            resized[df.index[i]] = image.reshape(-1)
        resized = pd.DataFrame(resized).T
        return resized

    def get_data(self):
        data = self.load_dataset()

        data = data.drop(['image_id'], axis=1).reset_index(drop=True)
        data = self.resize(data)/255
        data = data.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

        if self.name=='train':
            label = pd.read_csv("/content/gdrive/Shared drives/2020_Bengali/dataset/"
                                + self.name + ".csv")
            label = label.drop(['image_id', 'grapheme'], axis=1).reset_index(drop=True)
            x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size=0.2, random_state=42)
            del(data)
            del(label)
            return x_train, x_valid, y_train, y_valid
        elif self.name=='test':
            return data

