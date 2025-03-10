import requests
import zipfile
from tqdm import tqdm
import os

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        if 'Content-Length' in r.headers:
            total_size = int(r.headers['Content-Length'])
            pbar = tqdm(unit="B", total=total_size)
        else:
            total_size = None
            pbar = tqdm(unit="B")

        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename


def downloadextract(rawpath, name, link, redownload=False):
    zippath = os.path.join(rawpath, f"{name}.zip")
    targetpath = os.path.join(rawpath, f"{name}")
    if os.path.exists(targetpath) and redownload == False:
        print(f"{name} raw files already exist")
        return

    print(f"Downloading {name} files ...")
    download_file(link, zippath)

    print(f"Unzipping {name} files ...")
    with zipfile.ZipFile(zippath,"r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")