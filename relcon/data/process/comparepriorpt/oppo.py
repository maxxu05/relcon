"""
Preprocess Opportunity data into 30s and 10s windows
with a 15s and 5 sec overlap respectively
We use +- 16g format
Sample Rate: 30 Hz
Unit: milli g
https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition

For a typical challenge, 1 sec sliding window and 50%
overlap is used. Either specific sets or runs are used
for training or sometimes both run count and subject count are specified.



Usage:
    python oppo.py

"""

from scipy import stats as s
import numpy as np
import glob
import os
from tqdm import tqdm
from relcon.data.process.utils.download import downloadextract
from sklearn.model_selection import LeaveOneGroupOut

def main(rawpath):
    NAME = "opportunity"
    LINK = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"
    downloadextract(rawpath=rawpath, name=NAME, link = LINK)

    data_root = os.path.join(rawpath, NAME)
    data_path = os.path.join(data_root, "OpportunityUCIDataset/dataset/")
    file_paths = glob.glob(data_path + "*.dat")

    ##### prior code from original codebase: https://github.com/OxWearables/ssl-wearables
    print("Processing for 10sec window..")
    X_path, y_path, pid_path = get_write_paths(os.path.join(data_root, "processed"))
    epoch_len = 10
    overlap = 5
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)

    ##### now we slightly preprocess it further by creating files for each CV
    pid = np.load(os.path.join(data_root, "processed", "pid.npy"))
    X = np.load(os.path.join(data_root, "processed", "X.npy"))
    y = np.load(os.path.join(data_root, "processed", "Y.npy"))
    logo = LeaveOneGroupOut()
    inds = np.arange(pid.shape[0])
    for i, (train_inds, test_inds) in enumerate(logo.split(inds, groups=pid)):
        pid_train = pid[train_inds]

        logo2 = LeaveOneGroupOut()
        for _, (train_inds_true, val_inds) in enumerate(logo2.split(train_inds, groups=pid_train)):
            train_X = X[train_inds_true]
            train_y = y[train_inds_true]

            val_X = X[val_inds]
            val_y = y[val_inds]

            test_X = X[test_inds]
            test_y = y[test_inds]

            np.save(os.path.join(data_root, "processed", f"cv{i}_train_X.npy"), train_X)
            np.save(os.path.join(data_root, "processed", f"cv{i}_train_y.npy"), train_y)
            
            np.save(os.path.join(data_root, "processed", f"cv{i}_val_X.npy"), val_X)
            np.save(os.path.join(data_root, "processed", f"cv{i}_val_y.npy"), val_y)

            np.save(os.path.join(data_root, "processed", f"cv{i}_test_X.npy"), test_X)
            np.save(os.path.join(data_root, "processed", f"cv{i}_test_y.npy"), test_y)


def get_data_content(data_path):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(data_path).readlines()]
    datContent = np.array(datContent)

    label_idx = 243
    timestamp_idx = 0
    x_idx = 22
    y_idx = 23
    z_idx = 24
    index_to_keep = [timestamp_idx, label_idx, x_idx, y_idx, z_idx]
    # 3d +- 16 g

    datContent = datContent[:, index_to_keep]
    datContent = datContent.astype(float)
    datContent = datContent[~np.isnan(datContent).any(axis=1)]
    return datContent


def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):
    sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate)))

    sample_label_idx = 1
    sample_x_idx = 2
    sample_y_idx = 3
    sample_z_idx = 4

    sample_limit = sample_count * epoch_len * sample_rate
    data_content = data_content[:sample_limit, :]

    label = data_content[:, sample_label_idx]
    x = data_content[:, sample_x_idx]
    y = data_content[:, sample_y_idx]
    z = data_content[:, sample_z_idx]

    # to make overlappting window
    offset = overlap * sample_rate
    shifted_label = data_content[offset:-offset, sample_label_idx]
    shifted_x = data_content[offset:-offset:, sample_x_idx]
    shifted_y = data_content[offset:-offset:, sample_y_idx]
    shifted_z = data_content[offset:-offset:, sample_z_idx]

    shifted_label = shifted_label.reshape(-1, epoch_len * sample_rate)
    shifted_x = shifted_x.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y = shifted_y.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z = shifted_z.reshape(-1, epoch_len * sample_rate, 1)
    shifted_X = np.concatenate([shifted_x, shifted_y, shifted_z], axis=2)

    label = label.reshape(-1, epoch_len * sample_rate)
    x = x.reshape(-1, epoch_len * sample_rate, 1)
    y = y.reshape(-1, epoch_len * sample_rate, 1)
    z = z.reshape(-1, epoch_len * sample_rate, 1)
    X = np.concatenate([x, y, z], axis=2)

    X = np.concatenate([X, shifted_X])
    label = np.concatenate([label, shifted_label])
    return X, label


def clean_up_label(X, labels):
    # 1. remove rows with >50% zeros
    sample_count_per_row = labels.shape[1]

    rows2keep = np.ones(labels.shape[0], dtype=bool)
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == 0) > 0.5 * sample_count_per_row:
            rows2keep[i] = False

    labels = labels[rows2keep]
    X = X[rows2keep]

    # 2. majority voting for label in each epoch
    final_labels = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        final_labels.append(s.mode(row)[0])
    final_labels = np.array(final_labels, dtype=int)
    # print("Clean X shape: ", X.shape)
    # print("Clean y shape: ", final_labels.shape)
    return X, final_labels


def post_process_oppo(X, y, pid):
    zero_filter = np.array(y != 0)

    X = X[zero_filter]
    y = y[zero_filter]
    pid = pid[zero_filter]

    # change lie label from 5 to 3
    y[y == 5] = 3
    return X, y, pid


def process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap):
    X = []
    y = []
    pid = []
    sample_rate = 33

    for file_path in tqdm(file_paths):
        # print(file_path)
        subject_id = int(file_path.split("/")[-1][1:2])

        datContent = get_data_content(file_path)

        #############################################
        upsampledatContent = np.zeros((datContent.shape[0]*3, datContent.shape[1]))
        # import pdb; pdb.set_trace()

        upsampledatContent[0::3, 2:] = datContent[:, 2:]
        upsampledatContent[1:-2:3, 2:] = (datContent[:-1, 2:] + datContent[1:, 2:]) / 3
        upsampledatContent[2:-2:3, 2:] = (datContent[:-1, 2:] + datContent[1:, 2:]) * 2 / 3
        upsampledatContent[:, 0] = np.repeat(datContent[:, 0], 3)
        upsampledatContent[:, 1] = np.repeat(datContent[:, 1], 3)
        datContent = upsampledatContent
        sample_rate = 100
        #############################################

        current_X, current_y = content2x_and_y(
            datContent,
            sample_rate=sample_rate,
            epoch_len=epoch_len,
            overlap=overlap,
        )
        current_X, current_y = clean_up_label(current_X, current_y)
        if len(current_y) == 0:
            continue
        ids = np.full(
            shape=len(current_y), fill_value=subject_id, dtype=int
        )
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])

    # post-process
    y = y.flatten()
    X = X / 1000  # convert to g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)
    X, y, pid = post_process_oppo(X, y, pid)
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(pid_path, pid)


def get_write_paths(data_root):
    os.makedirs(data_root, exist_ok=True)
    X_path = os.path.join(data_root, "X.npy")
    y_path = os.path.join(data_root, "Y.npy")
    pid_path = os.path.join(data_root, "pid.npy")
    return X_path, y_path, pid_path


if __name__ == "__main__":
    main()