import argparse
import pickle
from datetime import date

import numpy as np
import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm.auto import tqdm
import zipfile

from relcon.data.process.utils.download import downloadextract

np.random.seed(42)

# ---------------------------------------------------------------------------------------------------------------------
def main(rawpath):
    NAME = "pamap2"
    LINK = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    downloadextract(rawpath=rawpath, name=NAME, link=LINK)
    data_root = os.path.join(rawpath, NAME)
    if not os.path.exists(os.path.join(data_root, NAME, "PAMAP2_Dataset")):
        with zipfile.ZipFile(
            os.path.join(data_root, "PAMAP2_Dataset.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(data_root)

    args = parse_arguments()
    args.dataset_loc = os.path.join(rawpath, NAME, "PAMAP2_Dataset")
    args.writefolder = os.path.join(rawpath, NAME, "processed")
    print(args)

    prepare_data(args)
    print("Data preparation complete!")


def most_common(lst):
    return max(set(lst.tolist()), key=lst.tolist().count)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for " "preparing PAMAP2")
    parser.add_argument(
        "--dataset_loc", type=str, default="", help="Location of the raw sensory data"
    )
    parser.add_argument(
        "--writefolder", type=str, default="", help="Location to write new files"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=100,
        help="Sampling rate for the data. Is used to "
        "downsample to the required rate",
    )
    parser.add_argument(
        "--original_sampling_rate",
        type=int,
        default=100,
        help="Original sampling rate for the dataset",
    )
    parser.add_argument(
        "--perform_normalization",
        type=str,
        default="True",
        help="To perform mean-variance normalization on the " "data",
    )
    parser.add_argument(
        "--num_sensor_channels",
        type=int,
        default=3,
        help="Number of sensor channels for used in the data " "preparation",
    )
    parser.add_argument(
        "--n_fold_validation",
        type=int,
        default=5,
        help="To extract data with n-folds instead of random "
        "20% test set. Default is 0, which creates the "
        "normal 80-20 split.",
    )

    args = parser.parse_args()

    return args


def map_activity_to_id(args):
    # List of activities being studied. Note that we *dont* use lying down class
    activity_list = [
        "lying",
        "sitting",
        "standing",
        "walking",
        "running",
        "cycling",
        "Nordic walking",
        "watching TV",
        "computer work",
        "car driving",
        "ascending stairs",
        "descending stairs",
        "vacuum cleaning",
        "ironing",
        "folding laundry",
        "house cleaning",
        "playing soccer",
        "rope jumping",
        "other (transient activities)",
    ]

    activity_id = {
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "running",
        6: "cycling",
        7: "Nordic walking",
        9: "watching TV",
        10: "computer work",
        11: "car driving",
        12: "ascending stairs",
        13: "descending stairs",
        16: "vacuum cleaning",
        17: "ironing",
        18: "folding laundry",
        19: "house cleaning",
        20: "playing soccer",
        24: "rope jumping",
        0: "other (transient activities)",
    }

    # chosen activity list
    chosen = {
        24: "rope jumping",
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "running",
        6: "cycling",
        7: "Nordic walking",
        12: "ascending stairs",
        13: "descending stairs",
        16: "vacuum cleaning",
        17: "ironing",
    }

    chosen_activity_list = [
        "rope jumping",
        "lying",
        "sitting",
        "standing",
        "walking",
        "running",
        "cycling",
        "Nordic walking",
        "ascending stairs",
        "descending stairs",
        "vacuum cleaning",
        "ironing",
    ]

    return chosen, chosen_activity_list


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(
        unique_subj, test_size=test_size, random_state=42
    )
    print("The train and validation subjects are: {}".format(train_val_subj))
    print("The test subjects are: {}".format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(
        train_val_subj, test_size=val_size, random_state=42
    )

    subjects = {"train": train_subj, "val": val_subj, "test": test_subj}

    return subjects


def get_acc_columns(col_start):
    # accelerometer
    sensor_col = np.arange(col_start + 1, col_start + 4)

    return sensor_col


def get_data(args):
    path_pamap2 = args.dataset_loc
    path_protocol = os.path.join(path_pamap2, "Protocol")
    path_optional = os.path.join(path_pamap2, "Optional")

    # The list of files in the protocol folder
    list_protocol = [
        os.path.join(path_protocol, item) for item in os.listdir(path_protocol)
    ]
    list_protocol.sort()

    # List of files in the optional folder
    list_optional = [
        os.path.join(path_optional, item) for item in os.listdir(path_optional)
    ]
    list_optional.sort()

    # Concatenating them together
    list_protocol += list_optional
    print(list_protocol)

    # Picking the sensor columns
    # Ankle IMU
    col_start = 37
    sensor_col = get_acc_columns(col_start)

    # Sanity check
    assert (
        len(sensor_col) == 3
    ), "Check the number of sensor channels obtained! Must be 3, is: {}".format(
        len(sensor_col)
    )

    # Placeholders to concatenate later
    sensor_all = np.empty((0, 3))
    target_all = np.empty((0,))
    subject_all = np.empty((0,))
    target_col = 1

    for prot in tqdm(list_protocol):
        data = np.loadtxt(prot)
        print("load from ...", prot, data.shape)

        assert data.shape[1] == 54

        # downsample
        interval = int(args.original_sampling_rate / args.sampling_rate)
        print("The interval is: {}".format(interval))
        idx_ds = np.arange(0, data.shape[0], interval)
        data = data[idx_ds]
        print("downsample ...", data.shape)

        # Get activity label
        target = data[:, target_col]
        print("target classes ...", np.unique(target))

        # handle NaN values
        sensor = np.array([pd.Series(i).interpolate() for i in data[:, sensor_col].T])

        # forward fill and backward fill missing values in the beginning and end
        sensor = np.array([pd.Series(i).fillna(method="ffill") for i in sensor])
        sensor = np.array([pd.Series(i).fillna(method="bfill") for i in sensor]).T
        assert np.all(np.isfinite(sensor)), "NaN in samples"
        assert sensor.shape[1] == len(sensor_col)

        # get subject
        basename = os.path.splitext(os.path.basename(prot))[0]
        assert basename[:-1] == "subject10"
        sID = int(basename[-1])
        subject = np.ones((target.shape[0],)) * sID

        # Concatenate
        sensor_all = np.concatenate((sensor_all, sensor), axis=0)
        target_all = np.concatenate((target_all, target), axis=0)
        subject_all = np.concatenate((subject_all, subject), axis=0)

    print("real {} {} {}".format(sensor_all.shape, target_all.shape, subject_all.shape))
    print("labels | real {}".format(np.unique(target_all)))
    print("subject | real {}".format(np.unique(subject_all)))

    # Putting it back into a dataframe
    df_cols = {"user": subject_all, "label": target_all}
    locs = ["ankle"]
    sensor_names = ["acc"]
    axes = ["x", "y", "z"]

    # Looping over all sensor locations
    count = 0
    sensor_col_names = []
    for loc in locs:
        for name in sensor_names:
            for axis in axes:
                c = loc + "_" + name + "_" + axis
                df_cols[c] = sensor_all[:, count]
                sensor_col_names.append(c)
                count += 1

    df = pd.DataFrame(df_cols)

    # Final size check
    assert (
        df.shape[1] == 5
    ), "All columns were not copied. " "Expected 5, got {}".format(df.shape[1])
    print("Done collecting! Shape is: {}".format(df.shape))

    # Removing some classes
    activity_id, activity_list = map_activity_to_id(args=args)

    df = df[df.label.isin(activity_id.keys())]
    print("After removing, the shape is: {}".format(df.shape))
    print("The activities are: {}".format(np.unique(df["label"])))

    # Need to encode the labels from 0:N-1 than what was available earlier
    le = LabelEncoder()
    encoded = le.fit_transform(df["label"].values)
    df["gt"] = encoded
    print("After label encoding, the shape is: {}".format(df.shape))
    print("The activities are: {}".format(np.unique(df["label"])))

    return df, sensor_col_names


def get_data_from_split(df, args, split, writefolder, n_fold=0):
    # Let us partition by train, val and test splits
    train_data = df[df["user"].isin(split["train"])]
    val_data = df[df["user"].isin(split["val"])]
    test_data = df[df["user"].isin(split["test"])]
    print(
        "The shapes of the splits are: {}, {} and {}".format(
            train_data.shape, val_data.shape, test_data.shape
        )
    )

    print("The unique classes in train are: {}".format(np.unique(train_data["label"])))
    print("The unique classes in val are: {}".format(np.unique(val_data["label"])))
    print("The unique classes in test are: {}".format(np.unique(test_data["label"])))

    # Choosing only the ankle accelerometry
    sensors = ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"]

    processed = {
        "train": {
            "data": train_data[sensors].values,
            "labels": train_data["gt"].values,
        },
        "val": {"data": val_data[sensors].values, "labels": val_data["gt"].values},
        "test": {"data": test_data[sensors].values, "labels": test_data["gt"].values},
        "fold": split,
    }

    # Sanity check on the sizes
    for phase in ["train", "val", "test"]:
        assert processed[phase]["data"].shape[0] == len(processed[phase]["labels"])

    for phase in ["train", "val", "test"]:
        print(
            "The phase is: {}. The data shape is: {}, {}".format(
                phase, processed[phase]["data"].shape, processed[phase]["labels"].shape
            )
        )

    # Before normalization
    print("Means before normalization")
    print(np.mean(processed["train"]["data"], axis=0))

    # # Creating logs by the date now. To make stuff easier
    # folder = os.path.join('all_data', date.today().strftime(
    #     "%b-%d-%Y"))
    # os.makedirs(folder, exist_ok=True)

    # os.makedirs(os.path.join(folder, 'unnormalized'), exist_ok=True)
    # args.n_fold = n_fold
    # if args.n_fold_validation != 0:
    #     save_name = 'pamap2_3_sr_{0.sampling_rate}_fold_{0.n_fold}' \
    #         .format(args)
    # else:
    #     save_name = 'pamap2_3_sr_{0.sampling_rate}'.format(args)

    # # Saving the joblib file
    # save_name += '.joblib'
    # name = os.path.join(folder, 'unnormalized', save_name)
    # with open(name, 'wb') as f:
    #     dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed["train"]["data"])
    # for phase in ['train', 'val', 'test']:
    #     processed[phase]['data'] = \
    #         scaler.transform(processed[phase]['data'])

    # After normalization
    print("Means after normalization")
    print(np.mean(processed["train"]["data"], axis=0))

    # # Saving into a joblib file
    # name = os.path.join(folder, save_name)
    # with open(name, 'wb') as f:
    #     dump(processed, f)

    # print('Saved into a joblib file!')

    ################################
    for split_iter in ["train", "val", "test"]:
        data_temp = []
        label_temp = []
        data = df[df["user"].isin(split[split_iter])]

        for User in tqdm(split[split_iter]):
            test_df_subject = data[data["user"] == User]

            for i in range(0, test_df_subject.shape[0] - 200, 200):
                activity = most_common(test_df_subject["gt"].iloc[i : i + 200])
                if activity == "nan":
                    continue
                label_temp.append(activity)
                temp = np.array(test_df_subject[sensors].iloc[i : i + 200])
                data_temp.append(scaler.transform(temp))

        data_temp = np.transpose(np.stack(data_temp), (0, 2, 1))
        print(data_temp.shape)
        label_temp = np.stack(label_temp)
        print(label_temp.shape)

        os.makedirs(writefolder, exist_ok=True)

        np.save(os.path.join(writefolder, f"cv{n_fold}_{split_iter}_X"), data_temp)
        np.save(os.path.join(writefolder, f"cv{n_fold}_{split_iter}_y"), label_temp)

    ################################

    return


def prepare_data(args):
    # Getting all the available data
    df, sensors = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df["user"].values)
    print("The unique subjects are: {}".format(unique_subj))

    # Performing the train-val-test split
    if args.n_fold_validation == 0:
        split = perform_train_val_test_split(unique_subj)
        get_data_from_split(
            df, split, args, writefolder=args.writefolder, n_fold=sensors
        )
    else:
        # TODO: define the 5-folds here
        n_fold_validation = 5
        num_test_subj = int(np.ceil((1.0 / n_fold_validation) * len(unique_subj)))
        print("The number of validation and test subjects: " "{}".format(num_test_subj))

        sanity = {"train": [], "val": [], "test": []}

        for i in tqdm(range(n_fold_validation)):
            # First fold is same as random 80:20 split
            if i == 0:
                split = perform_train_val_test_split(unique_subj)
                train_subj = split["train"]
                val_subj = split["val"]
                test_subj = split["test"]
            else:
                remaining_test = list(set(unique_subj) - set(sanity["test"]))

                # Going to shuffle it in place and pick the first num_test_subj
                np.random.shuffle(remaining_test)

                if i != n_fold_validation - 1:
                    test_subj = remaining_test[:num_test_subj]
                else:
                    test_subj = remaining_test

                # Remaining participants for train+val
                train_val = list(set(unique_subj) - set(test_subj))

                # Splitting that 80:20
                train_subj, val_subj = train_test_split(
                    train_val, test_size=0.2, random_state=42
                )

            # Sanity check to make sure all subjects were in test/val
            # once only
            sanity["train"].extend(train_subj)
            sanity["val"].extend(val_subj)
            sanity["test"].extend(test_subj)

            # assert len(test_subj) == 2

            subjects = {"train": train_subj, "val": val_subj, "test": test_subj}
            print(i, subjects)

            # Saving the split data
            get_data_from_split(
                df, args, split=subjects, writefolder=args.writefolder, n_fold=i
            )

        # For test split, there have to be each participant only once
        print(sanity)
        assert len(sanity["test"]) == 9
        assert sanity["test"].sort() == list(unique_subj).sort()

        v, c = np.unique(sanity["test"], return_counts=True)
        assert np.sum(c == 1) == 9

    return
