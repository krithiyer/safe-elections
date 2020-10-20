# This file was adopted from https://github.com/huggingface/transformers and then modified.
# Refer to that repository for the copyright notice.

""" Script for downloading GLUE data.
"""

import argparse
import os
import sys
import urllib.request
import zipfile

TASK2PATH = {
    "QQP": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5",
}

def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="directory to save data to", type=str, default="glue_data")
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    download_and_extract('QQP', args.data_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))