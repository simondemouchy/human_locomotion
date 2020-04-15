import json
import os
import tarfile
from os.path import join as pjoin

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


PATH_DATA = pjoin(os.path.dirname(os.path.abspath(__file__)), "data")
ARCHIVE_FNAME = "public.tar.gz"
DOWNLOAD_URL = "https://plmbox.math.cnrs.fr/f/8224e749026747758c56/?dl=1"
RANDOM_STATE = 777


def download_data():
    """This function downloads the data, extracts them and removes the archive."""
    if not os.path.exists(PATH_DATA):
        print("Data are missing. Downloading them now (~150 Mo)...",
              end="", flush=True)
        urlretrieve(DOWNLOAD_URL, ARCHIVE_FNAME)
        print("Ok.")
        print("Extracting now...", end="", flush=True)
        tf = tarfile.open(ARCHIVE_FNAME)
        tf.extractall()
        print("Ok.")
        print("Removing the archive...", end="", flush=True)
        os.remove(ARCHIVE_FNAME)
        print("Ok.")
        os.rename("public", "data")


if __name__ == "__main__":
    download_data()
