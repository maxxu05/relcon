import numpy as np
import os
from tqdm import tqdm

PATH = "relcon/data/datasets/dummydataset"
NUM_SUBJECTS = 64
NUM_HOURS_PER_SUBJECT = 2
NUM_TS_PER_HOUR = 25

TIMELEN = 256  # 2.56 long sequence sampled at 100 Hz
CHANNELS = 3  # 3 axis accerlometry


def main():
    # if os.path.exists(PATH):
    #     print("dataset already exists")
    #     return

    os.makedirs(PATH, exist_ok=True)
    for subject_id in tqdm(range(NUM_SUBJECTS)):
        # construct parent folder for train/val/test
        if subject_id < NUM_SUBJECTS // 2:
            TYPE = "train"
        elif subject_id < 3* NUM_SUBJECTS // 4:
            TYPE = "val"
        else:
            TYPE = "test"
        typepath = os.path.join(PATH, TYPE)
        os.makedirs(typepath, exist_ok=True)

        # construct sub-parent folder for the subject-level
        subjectpath = os.path.join(typepath, f"subject_{subject_id}")
        os.makedirs(subjectpath, exist_ok=True)
        for hour_id in range(NUM_HOURS_PER_SUBJECT):
            # Construct child folder for hour-level
            # The other time-series alongside the anchor time-series within the folder form the "within-user" candidates
            hourpath = os.path.join(subjectpath, f"hour_{hour_id}")
            os.makedirs(hourpath, exist_ok=True)

            # construct 2.56 second chunks sampled at 100hz
            for i in range(NUM_TS_PER_HOUR):
                # models expect numpy arrays of size TIMELEN, CHANNELS
                timeseries = np.random.normal(size=(TIMELEN, CHANNELS))
                np.save(os.path.join(hourpath, f"ts_{i}"), timeseries)


if __name__ == "__main__":
    main()
