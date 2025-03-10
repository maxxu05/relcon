import argparse
from datetime import date

import numpy as np
import os
import pandas as pd
import random
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import zipfile

from relcon.data.process.utils.download import downloadextract

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------------------------------------------------
def main(rawpath):
    NAME = "motionsense"
    os.makedirs(os.path.join(rawpath, NAME), exist_ok=True)
    LINK1 = "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/B_Accelerometer_data.zip"
    downloadextract(rawpath=os.path.join(rawpath, NAME), name="B_Accelerometer_data", link = LINK1)
    LINK2 = "https://github.com/mmalekzadeh/motion-sense/raw/refs/heads/master/data/C_Gyroscope_data.zip"
    downloadextract(rawpath=os.path.join(rawpath, NAME), name="C_Gyroscope_data", link = LINK2)
    # data_root = os.path.join(rawpath, NAME)
    # if not os.path.exists(os.path.join(data_root, NAME, "Activity recognition exp")):
    #     with zipfile.ZipFile(
    #         os.path.join(data_root, "Activity recognition exp.zip"), "r"
    #     ) as zip_ref:
    #         zip_ref.extractall(data_root)

    args = parse_arguments()
    args.dataset_loc = os.path.join(rawpath, NAME)
    args.writefolder = os.path.join(rawpath, NAME, "processed")
    print(args)

    prepare_data(args)
    print('Data preparation complete!')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters for '
                                                 'preparing Motionsense')
    parser.add_argument('--dataset_loc', type=str,
                        default='/coc/pcba1/hharesamudram3/data_preparation/'
                                'all_data/motionsense/data/',
                        help='Location of the raw sensory data')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        help='Sampling rate for the data. Is used to '
                             'downsample to the required rate')
    parser.add_argument('--perform_normalization', type=str, default='True',
                        help='To perform mean-variance normalization on the '
                             'data')
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
    
# from the dataset repo
def get_ds_infos():
    """
    Read the file includes data subject information.

    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]

    Returns:
        A pandas DataFrame that contains inforamtion about data subjects'
        attributes
    """

    dss = pd.read_csv("../data/data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")

    return dss


def map_activity_to_id():
    action_to_id = {"dws": 0, "ups": 1, "wlk": 2, "jog": 3, "std": 4, "sit": 5}
    folder_to_action = {'dws_1': 0, 'dws_11': 0, 'dws_2': 0, 'jog_16': 3,
                        'jog_9': 3, 'sit_13': 5, 'sit_5': 5,
                        'std_14': 4, 'std_6': 4, 'ups_12': 1, 'ups_3': 1,
                        'ups_4': 1, 'wlk_15': 2, 'wlk_7': 2,
                        'wlk_8': 2}

    return action_to_id, folder_to_action


def read_data(current, acc_folder, gyro_folder, args):
    all_data = np.zeros((1, 7))

    for subj in range(1, 25):
        acc = os.path.join(args.root_folder, acc_folder, current,
                           'sub_' + str(subj) + '.csv')
        acc_data = pd.read_csv(acc)

        gyro = os.path.join(args.root_folder, gyro_folder, current,
                            'sub_' + str(subj) + '.csv')
        gyro_data = pd.read_csv(gyro)

        # Dropping the first column
        acc_data = acc_data.drop(['Unnamed: 0'], axis=1)
        gyro_data = gyro_data.drop(['Unnamed: 0'], axis=1)

        # Matching the number of lines from the accelerometer and gyroscope
        acc_data = acc_data.iloc[:len(gyro_data), :]

        # Taking just the data
        gyro_data = gyro_data.values
        acc_data = acc_data.values

        # Stacking them together
        data = np.hstack((acc_data, gyro_data))

        # Sampling at the given sampling rate
        # 50 HZ IS THE BASE RATE USED IN THE DATASET
        # divide_by = np.int(np.round(50 / args.sampling_rate))
        # data = data[np.arange(0, len(data), divide_by), :]

        ################## new sampling rates ##########################
        upsampled = np.zeros((data.shape[0]*2, data.shape[1]))
        upsampled[0::2, :] = data
        # import pdb; pdb.set_trace()
        upsampled[1:-1:2, :] = (data[:-1, :] + data[1:, :]) / 2
        upsampled[-1, :] = data[-1, :]
        data = upsampled
        
        ################## new sampling rates ##########################

        # Adding in the subject IDs:
        subj_id = np.ones((len(data), 1)) * subj

        # Subj_ID + data
        subj_id_data = np.hstack((subj_id, data))

        all_data = np.vstack((all_data, subj_id_data))

    all_data = all_data[1:, :]
    #     print('Cumulative size: {}'.format(all_data.shape))
    return all_data


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj,
                                                 test_size=test_size,
                                                 random_state=42)
    print('The train and validation subjects are: {}'.format(train_val_subj))
    print('The test subjects are: {}'.format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj,
                                            test_size=val_size,
                                            random_state=42)

    subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}

    return subjects


def get_data_from_split(df, args, split, n_fold=0):
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
    #     save_name = 'motionsense_{0.num_sensor_channels}_sr_' \
    #                 '{0.sampling_rate}_fold_{0.n_fold}.joblib'.format(args)
    # else:
    #     save_name = 'motionsense_{0.num_sensor_channels}_sr_{0.sampling_rate}' \
    #                 '.joblib'.format(args)

    # name = os.path.join(folder, 'unnormalized', save_name)
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

    return


def get_data(args):
    acc_folder = 'B_Accelerometer_data/B_Accelerometer_data'
    gyro_folder = 'C_Gyroscope_data/C_Gyroscope_data'
    args.root_folder = args.dataset_loc

    # Listing all the sub-folders inside both acc_folder and gyro_folder
    acc_path = os.path.join(args.root_folder, acc_folder)
    print(os.listdir(acc_path))

    gyro_path = os.path.join(args.root_folder, gyro_folder)
    print(os.listdir(gyro_path))

    # Let us print the number of subjects in each sub-folder
    all_folders = os.listdir(acc_path)

    _, f_to_a = map_activity_to_id()

    # The dataframe which we will use to store the data into
    cols = {'user': [], 'acc_x': [], 'acc_y': [], 'acc_z': [], 'gyro_x': [],
            'gyro_y': [], 'gyro_z': [], 'gt': []}
    df = pd.DataFrame(cols)
    df.head()

    # Starting the loop
    all_data = np.zeros((1, 8))
    for i in tqdm(range(0, len(all_folders))):
        # print('In the folder: {}'.format(all_folders[i]))
        current = all_folders[i]

        # Reading in the data
        data = read_data(current, acc_folder, gyro_folder, args)

        # Label
        label = f_to_a[current]
        labels = np.ones((len(data), 1)) * label

        # Stacking them
        both = np.hstack((data, labels))

        all_data = np.vstack((all_data, both))
        print(all_data.shape)

    # Adding to the data frame
    df['user'] = all_data[1:, 0]
    df['acc_x'] = all_data[1:, 1]
    df['acc_y'] = all_data[1:, 2]
    df['acc_z'] = all_data[1:, 3]
    df['gyro_x'] = all_data[1:, 4]
    df['gyro_y'] = all_data[1:, 5]
    df['gyro_z'] = all_data[1:, 6]
    df['gt'] = all_data[1:, 7]

    print('Done collecting!')

    return df


def prepare_data(args):
    # Loading in all the data first
    df = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df['user'].values)
    print('The unique subjects are: {}'.format(unique_subj))

    # Performing the train-val-test split
    if args.n_fold_validation == 0:
        split = perform_train_val_test_split(unique_subj)
        get_data_from_split(df, args=args, split=split)
    else:
        n_fold_validation = args.n_fold_validation
        num_test_subj = int(np.ceil((1.0 / n_fold_validation) *
                                    len(unique_subj)))
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

            # Sanity check to make sure all subjects were in test/val
            # once only
            sanity['train'].extend(train_subj)
            sanity['val'].extend(val_subj)
            sanity['test'].extend(test_subj)

            assert len(test_subj) == 5 if i != n_fold_validation - 1 else 4

            subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}
            print(i, subjects)

            # Saving the split data
            get_data_from_split(df, args, split=subjects, n_fold=i)

        # For test split, there have to be each participant only once
        print(sanity)
        assert len(sanity['test']) == 24

        v, c = np.unique(sanity['test'], return_counts=True)
        assert np.sum(c == 1) == 24

    return
