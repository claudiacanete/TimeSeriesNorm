r"""
File to extract csv images from csv files for mnist dataset.
"""

import os
import cv2
from tqdm import tqdm
import numpy as np
import _csv as csv
import yfinance as yf
import pandas as pd
import yaml
import argparse
import math
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Arguments for ddpm training')
parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
args = parser.parse_args()
with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
model_config = config['model_params']           
horizon=model_config['time_horizon']

chunks = pd.read_csv('data/norm.csv') 
# Convert chunks to a DataFrame
df_chunks = pd.DataFrame(chunks)
# Remov
df_resized = df_chunks.apply(lambda row: row[:65], axis=1)

df_resized.iloc[:, 0] = '1'

lista=list(range(0,65))
df_resized.columns = lista

df_resized.index = df_resized.index + 1
df_resized.sort_index(inplace=True)

print(df_resized.shape)
print(df_resized.head())
df_resized.to_csv('data/norm.csv', index=False)



def extract_images(save_dir, csv_fname):
    assert os.path.exists(save_dir), "Directory {} to save images does not exist".format(save_dir)
    assert os.path.exists(csv_fname), "Csv file {} does not exist".format(csv_fname)
    with open(csv_fname) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            im = np.zeros((horizon))
            im[:] = list(row[1:])
            
            #size_image=int(math.sqrt(horizon))
            size_image=horizon
            im = im.reshape((1,size_image))
            if not os.path.exists(os.path.join(save_dir, row[0])):
                os.mkdir(os.path.join(save_dir, row[0]))
            cv2.imwrite(os.path.join(save_dir, row[0], '{}.png'.format(idx)), im)
            if idx % 1000 == 0:
                print('Finished creating {} images in {}'.format(idx+1, save_dir))     



if __name__ == '__main__':
 extract_images('data/train/images', 'data/norm.csv')
 #extract_images('data/test/images', 'data/test.csv')