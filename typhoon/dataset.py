import os
import torch
import torch.utils.data as Data
import glob
import numpy as np
from scipy.io import loadmat
from utils import *
from typhoon.model import *


def physical_dataset():
    mat_dir = r'I:\mat'
    mat_sub_dirs = [os.path.join(mat_dir,mat_sub_dir) for mat_sub_dir in os.listdir(mat_dir)]
    # for mat_sub_dir in mat_sub_dirs:
    #     name = glob.glob(os.path.join(mat_sub_dir,'*.mat'))
    #     a = glob.glob(mat_dir+'/*/'+'1979010106.mat')+glob.glob(mat_dir+'/*/*/'+'1979010106.mat')
    #     print(name)

    mat = loadmat(r'I:\mat\mat_sm_u\1979010106.mat')
    mat = mat['mat']
    print()

def location_dataset(SEQUENCE_LENGTH=8,TYPHOON_MIN_LENGTH=20):
    time_location_train = load_data(r'F:\mytoys\python\dataset\train.txt', long=TYPHOON_MIN_LENGTH)
    time_location_test = load_data(r'F:\mytoys\python\dataset\test.txt', long=TYPHOON_MIN_LENGTH)

    location_train_typhoon = np.array(del_time(time_location_train))
    location_test_typhoon = np.array(del_time(time_location_test))

    location_train = organize_data(location_train_typhoon,SEQUENCE_LENGTH)
    location_test = organize_data(location_test_typhoon,SEQUENCE_LENGTH)

    location_train_data = location_train[:,:SEQUENCE_LENGTH,:2]
    location_train_label = location_train[:,SEQUENCE_LENGTH:,:2]
    location_test_data = location_test[:,:SEQUENCE_LENGTH,:2]
    location_test_label = location_test[:,SEQUENCE_LENGTH:,:2]
    return location_train_data, location_train_label.astype(float), location_test_data, location_test_label.astype(float)

loc_train_data, loc_train_label, loc_test_data, loc_test_label\
    = location_dataset()

# list
# [result_n,
# lat_max, lat_min,
# lon_max, lon_min,
# pre_max, pre_min,
# lat_mean, lon_mean, pre_mean]
location_train_data_norm = normalize(loc_train_data.astype(float))
location_train_label_norm = normalize(loc_train_label.astype(float))
location_test_data_norm = normalize(loc_test_data.astype(float))
location_test_label_norm = normalize(loc_test_label.astype(float))

# 归一化后的 lat,lon,pressure
location_train_data, location_train_label, location_test_data, location_test_label\
    = torch.from_numpy(location_train_data_norm[0]).float(), \
      torch.from_numpy(location_train_label_norm[0]).float(), \
      torch.from_numpy(location_test_data_norm[0]).float(), \
      torch.from_numpy(location_test_label_norm[0]).float()

# location_train = Data.TensorDataset(location_train_data,
#                                     location_train_label)
# float64 --> float32
location_train = Data.TensorDataset(location_train_data,
                                    location_train_label)

train_loader = Data.DataLoader(dataset=location_train,batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=2,
                               drop_last=True)
