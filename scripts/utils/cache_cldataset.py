import argparse
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
import glob
from tqdm import tqdm
import subprocess
import filelock
import sys
import random

sys.path.insert(1, os.getcwd())
from solo.data.cldataset import H5Dataset


def test_load(directory, dataset, full=False):
    try:
        # add pwd to python path
        dataset = H5Dataset(
            directory=directory, 
            dataset=dataset,
            partition="train"
        )
        print(f"Dataset length: {len(dataset)}")
        if full:
            for i in tqdm(range(len(dataset))):
                dataset[i]

        else:
            N = 1000
            print(f"check first {N} samples")
            for i in tqdm(range(N)):
                dataset[i]

            print(f"check {N} random samples")
            for i in tqdm(random.sample(range(len(dataset)), N)):
                dataset[i]

            print(f"check last {N} samples")
            for i in tqdm(range(N)):
                dataset[-i]
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    print("Test passed")
    return True


def unzip_data_files(src_directory: str, out_directory_root: str) -> None:
    """
    Extracts the contents of zip files in a directory into nested folders.

    Args:
        directory: The path to the directory containing the zip files.

    Returns:
        None
    """

    zip_files = [zip_file for zip_file in os.listdir(src_directory) if zip_file.endswith(".zip")]

    os.makedirs(out_directory_root, exist_ok=True)

    def extract_single_zip(zip_file: str) -> None:

        zip_path = os.path.join(src_directory, zip_file)
        output_dir = os.path.join(
            out_directory_root, os.path.splitext(os.path.basename(zip_file))[0])

        os.makedirs(output_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    with ThreadPoolExecutor() as executor, tqdm(total=len(zip_files)) as pbar:
        futures_list = []
        for zip_file in zip_files:
            future = executor.submit(extract_single_zip, zip_file)
            future.add_done_callback(lambda p: pbar.update(1))
            futures_list.append(future)

        # Wait for all tasks to complete
        for future in futures_list:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache CGLM dataset files.")
    parser.add_argument(
        "--src_directory", type=str, default="datasets/cldatasets/", help="The directory where the dataset files are.")
    parser.add_argument(
        "--out_directory_root", type=str, default="/tmp/cldatasets/", help="The directory where the dataset will be extracted.")
    parser.add_argument(
        "--dataset", type=str, default="CGLM", help="The name of the dataset to download.")
    parser.add_argument(
        "--full_check", action="store_true", help="Whether to perform a full check of the dataset files before extracting.")
    args = parser.parse_args()


    start = time.time()
    os.makedirs(args.out_directory_root, exist_ok=True)
    

    with filelock.FileLock(f"{args.out_directory_root}/lock"):
        success = test_load(
            directory=os.path.join(args.out_directory_root, f"{args.dataset}"),
            dataset=args.dataset,
            full=args.full_check
        )
        if success:
            print("Dataset can be loaded. Skipping cache.")
        else:
            print("Creating new cache.")

            # copy the metadata file
            # os.system(f"cp -rv {args.src_directory}/order_files {args.out_directory_root}/order_files")
            try:
                os.makedirs(f"{args.out_directory_root}/{args.dataset}/order_files", exist_ok=True)
                for file in glob.glob(f"{args.src_directory}/{args.dataset}/order_files/*"):
                    subprocess.check_output([
                        "cp", "-v", 
                        file, 
                        f"{args.out_directory_root}/{args.dataset}/order_files/"
                    ])
                
            except subprocess.CalledProcessError as e:
                raise e

            # Extract and cache the dataset files
            unzip_data_files(
                args.src_directory+f"/{args.dataset}/data",
                args.out_directory_root+f"/{args.dataset}/data"
            )

            end = time.time()
            print(f"Time taken to cache CGLM dataset files: {end-start} seconds.")
