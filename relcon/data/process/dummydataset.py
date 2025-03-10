import numpy as np
import os
from tqdm import tqdm

PATH = "relcon/data/datasets/dummydataset"
NUM_SUBJECTS = 1000
NUM_HOURS_PER_SUBJECT = 2
TIMELEN = 256  # 2.56 long sequence sampled at 100 Hz
CHANNELS = 3  # 3 axis accerlometry


def main():
    # if os.path.exists(PATH):
    #     print("dataset already exists")
    #     return

    os.makedirs(PATH, exist_ok=True)
    for subject_id in tqdm(range(NUM_SUBJECTS)):
        # construct parent folder for the subject-level
        subjectpath = os.path.join(PATH, f"subject_{subject_id}")
        os.makedirs(subjectpath, exist_ok=True)
        for hour_id in range(NUM_HOURS_PER_SUBJECT):
            # Construct child folder for hour-level
            # The other time-series alongside the anchor time-series within the folder form the "within-user" candidates
            hourpath = os.path.join(subjectpath, f"hour_{hour_id}")
            os.makedirs(hourpath, exist_ok=True)

            # construct 2.56 second chunks in a 100hz 1 hour chunk
            for i in range(0, 100 * 60 * 60, 256):
                # models expect numpy arrays of size TIMELEN, CHANNELS
                timeseries = np.random.normal(size=(TIMELEN, CHANNELS))
                np.save(os.path.join(hourpath, f"ts_{i}"), timeseries)


if __name__ == "__main__":
    main()
