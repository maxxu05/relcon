import argparse
import pickle
from datetime import date
from tqdm import tqdm

import numpy as np
import os
import pandas as pd
import random
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile

from relcon.data.process.utils.download import downloadextract

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------------------------------------------------
def main(rawpath):
    NAME = "hhar"
    LINK = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
    downloadextract(rawpath=rawpath, name=NAME, link = LINK)
    data_root = os.path.join(rawpath, NAME)
    if not os.path.exists(os.path.join(data_root, NAME, "Activity recognition exp")):
        with zipfile.ZipFile(
            os.path.join(data_root, "Activity recognition exp.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(data_root)

    args = parse_arguments()
    args.dataset_loc = os.path.join(rawpath, NAME, "Activity recognition exp", "Watch_accelerometer.csv")
    args.writefolder = os.path.join(rawpath, NAME, "processed")
    print(args)

    prepare_data(args)
    print('Data preparation complete!')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters for '
                                                 'preparing HHAR watch')
    parser.add_argument('--dataset_loc', type=str,
                        default='/coc/pcba1/hharesamudram3/capture_24/code'
                                '/data_preparation/hhar/data/hhar/'
                                'Activity recognition exp/'
                                'Watch_accelerometer.csv',
                        help='Location of the raw sensory data')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        help='Sampling rate for the data. Is used to '
                             'downsample to the required rate')
    parser.add_argument('--num_sensor_channels', type=int, default=3,
                        help='Number of sensor channels for used in the data '
                             'preparation')
    parser.add_argument('--n_fold_validation', type=int, default=5,
                        help='To extract data with n-folds instead of random '
                             '20% test set. Default is 0, which creates the '
                             'normal 80-20 split.')

    args = parser.parse_args()

    return args

def most_common(lst):
    return max(set(lst.tolist()), key=lst.tolist().count)
    
def map_activity_to_id():
    # List of activities being studied. Note that we *dont* use lying down class
    activity_list = ['bike', 'sit', 'stairsdown', 'stairsup', 'stand', 'walk']

    activity_id = {'bike': 0, 'sit': 1, 'stairsdown': 2, 'stairsup': 3,
                   'stand': 4, 'walk': 5}

    return activity_id, activity_list


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj,
                                                 test_size=test_size,
                                                 random_state=42)
    print('The train and validation subjects are: {}'.format(train_val_subj))
    print('The test subjects are: {}'.format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj, test_size=val_size,
                                            random_state=42)

    subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}

    return subjects


def get_data(args):
    # Getting the activity labels
    activity_id, activity_list = map_activity_to_id()

    # Loading data from the csv
    data = pd.read_csv(args.dataset_loc)
    data = data.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------------
    # Downsampling based on the recording device.
    # Gear=100Hz, LGWatch=200Hz # as per the original paper:
    # https://pure.au.dk/ws/files/93103132/sen099_stisenAT3.pdf
    gear = data[data['Model'] == 'gear']
    lg = data[data['Model'] == 'lgwatch']
    print('The gear watch data: {}, LG Watch data: {}'.format(gear.shape,
                                                              lg.shape))

    # downsample to 100 hz
    # divide_by = int(np.round(100 / 50))
    # index = np.arange(0, len(gear), divide_by)
    # gear = gear.iloc[index, :].reset_index(drop=True)

    divide_by = int(np.round(200 / 100))
    index = np.arange(0, len(lg), divide_by)
    lg = lg.iloc[index, :].reset_index(drop=True)

    data = pd.concat((gear, lg))
    print('Post down-sampling, the gear watch data: {}, LG Watch data: {}.'
          'Total size: {}'.format(gear.shape, lg.shape, data.shape))

    df = pd.DataFrame()
    df['acc_x'] = data['x']
    df['acc_y'] = data['y']
    df['acc_z'] = data['z']
    df['gt'] = data['gt'].map(activity_id)

    # participant_to_id = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7,
    #                      'h': 8, 'i': 9}
    df['user'] = data['User']

    print('Done collecting!')
    return df


def get_data_from_split(df, split, args, n_fold=0):
    # Let us partition by train, val and test splits
    train_data = df[df['user'].isin(split['train'])]
    val_data = df[df['user'].isin(split['val'])]
    test_data = df[df['user'].isin(split['test'])]
    print('The shapes of the splits are: {}, {} and {}'.
          format(train_data.shape, val_data.shape, test_data.shape))

    print('The unique classes in train are: {}'
          .format(np.unique(train_data['gt'])))
    print('The unique classes in val are: {}'
          .format(np.unique(val_data['gt'])))
    print('The unique classes in test are: {}'
          .format(np.unique(test_data['gt'])))

    if args.num_sensor_channels == 3:
        sensors = ['acc_x', 'acc_y', 'acc_z']
    elif args.num_sensor_channels == 6:
        sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

        
    processed = {'train': {'data': train_data[sensors].values,
                           'labels': train_data['gt'].values},
                 'val': {'data': val_data[sensors].values,
                         'labels': val_data['gt'].values},
                 'test': {'data': test_data[sensors].values,
                          'labels': test_data['gt'].values},
                 'fold': split
                 }

    # Sanity check on the sizes
    for phase in ['train', 'val', 'test']:
        assert processed[phase]['data'].shape[0] == \
               len(processed[phase]['labels'])

    for phase in ['train', 'val', 'test']:
        print('The phase is: {}. The data shape is: {}, {}'
              .format(phase, processed[phase]['data'].shape,
                      processed[phase]['labels'].shape))

    # Before normalization
    print('Means before normalization')
    print(np.mean(processed['train']['data'], axis=0))

    # Creating logs by the date now. To make stuff easier
    # folder = os.path.join('all_data', date.today().strftime(
    #     "%b-%d-%Y"))
    # os.makedirs(folder, exist_ok=True)

    # os.makedirs(os.path.join(folder, 'unnormalized'), exist_ok=True)
    # args.n_fold = n_fold
    # if args.n_fold_validation != 0:
    #     save_name = 'hhar_watch_sr_{0.sampling_rate}_fold_{0.n_fold}' \
    #         .format(args)
    # else:
    #     save_name = 'hhar_watch_sr_{0.sampling_rate}'.format(args)

    # Saving the joblib file
    # save_name += '.joblib'
    # name = os.path.join(folder, 'unnormalized', save_name)
    # import pdb; pdb.set_trace()
    # with open(name, 'wb') as f:
    #     dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed['train']['data'])


    # After normalization
    print('Means after normalization')
    print(np.mean(processed['train']['data'], axis=0))

    # # Saving into a joblib file
    # name = os.path.join(folder, save_name)
    # with open(name, 'wb') as f:
    #     dump(processed, f)

    # print('Saved into a joblib file!')

    
    ################################ 
    for split_iter in ["train", "val", "test"]:
        data_temp = []
        label_temp = []
        data = df[df['user'].isin(split[split_iter])]
        
        for User in tqdm(split[split_iter]):
            test_df_subject = data[data["user"] == User]
        
            for i in range(0, test_df_subject.shape[0]-200, 200):
                activity = most_common(test_df_subject["gt"].iloc[i:i+200])
                if activity == "nan":
                    continue
                label_temp.append(activity)
                temp = np.array(test_df_subject[sensors].iloc[i:i+200])
                data_temp.append(scaler.transform(temp))

        data_temp = np.transpose(np.stack(data_temp), (0,2,1))
        print(data_temp.shape)
        label_temp = np.stack(label_temp)
        print(label_temp.shape)

        os.makedirs(args.writefolder, exist_ok=True)

        np.save(os.path.join(args.writefolder, f'cv{n_fold}_{split_iter}_X'), data_temp)
        np.save(os.path.join(args.writefolder,f"cv{n_fold}_{split_iter}_y"), label_temp)
        
    ################################
    


    # return


def prepare_data(args):
    # Reading in all the data
    df = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df['user'].values)
    print('The unique subjects are: {}'.format(unique_subj))

    # Performing the train-val-test split
    if args.n_fold_validation == 0:
        split = perform_train_val_test_split(unique_subj)
        get_data_from_split(df, split, args)
    else:
        n_fold_validation = args.n_fold_validation
        num_test_subj = int(np.ceil((1.0 / n_fold_validation)
                                     * len(unique_subj)))
        print('The number of validation and test subjects: '
              '{}'.format(num_test_subj))

        sanity = {'train': [], 'val': [], 'test': []}

        for i in range(n_fold_validation):
            # First fold is same as random 80:20 split
            if i == 0:
                split = perform_train_val_test_split(unique_subj)
                train_subj = split['train']
                val_subj = split['val']
                test_subj = split['test']
            else:
                remaining_test = list(set(unique_subj) - set(sanity['test']))

                # Going to shuffle it in place and pick the first num_test_subj
                np.random.shuffle(remaining_test)

                if i != n_fold_validation - 1:
                    test_subj = remaining_test[:num_test_subj]
                else:
                    test_subj = remaining_test

                # Remaining participants for train+val
                train_val = list(set(unique_subj) - set(test_subj))

                # Splitting that 80:20
                train_subj, val_subj = train_test_split(train_val,
                                                        test_size=0.2,
                                                        random_state=42)

            # Sanity check to make sure all subjects were in test/val once only
            sanity['train'].extend(train_subj)
            sanity['val'].extend(val_subj)
            sanity['test'].extend(test_subj)

            subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}
            print(i, subjects)

            get_data_from_split(df, split=subjects, args=args, n_fold=i)

        # For test split, there have to be each participant only
        # once
        print(sanity)
        assert len(sanity['test']) == 9

        v, c = np.unique(sanity['test'], return_counts=True)
        assert np.sum(c == 1) == 9

    return
