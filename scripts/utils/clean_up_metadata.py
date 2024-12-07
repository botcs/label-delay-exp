import os
import torch
import argparse
from tqdm import tqdm
from multiprocessing import Pool

# use argparse to get the paths
parser = argparse.ArgumentParser()
parser.add_argument("--img_root", type=str, default="./datasets/cloc/images")
parser.add_argument("--img_paths_fname", type=str, default="./datasets/cloc/release/train_store_loc.torchSave")
parser.add_argument("--num_images", "-N", type=int, default=-1, help="Number of images to keep (default: -1, keep all images)")
# 30000*128 = 3840000


args = parser.parse_args()

print("img_root:", args.img_root)
print("img_paths_fname:", args.img_paths_fname)
print("num_images:", args.num_images)

# Check if the reduced pathfile already exists
binary_root = os.path.dirname(args.img_paths_fname)
reduced_img_paths_fname = binary_root + f"/train_{args.num_images}_filenames.txt"
reduced_img_paths_bin_fname = binary_root + f"/train_{args.num_images}_store_loc.torchSave"
reduced_labels_fname = binary_root + f"/train_{args.num_images}_labels.torchSave"
reduced_time_fname = binary_root + f"/train_{args.num_images}_time.torchSave"
reduced_user_fname = binary_root + f"/train_{args.num_images}_userID.torchSave"



# load the image paths
print("Loading image paths...")
img_paths_bin = torch.load(args.img_paths_fname)

if args.num_images == -1:
    args.num_images = len(img_paths_bin)

img_paths_bin = img_paths_bin[:int(args.num_images*1.1)]
labels = torch.load(binary_root + "/train_labels.torchSave")[:int(args.num_images*1.1)]
times = torch.load(binary_root + "/train_time.torchSave")[:int(args.num_images*1.1)]
user_IDs = torch.load(binary_root + "/train_userID.torchSave")[:int(args.num_images*1.1)]

def exists(path):
    abs_path = args.img_root + "/" + path
    abs_path = abs_path.strip()
    return abs_path if os.path.exists(abs_path) else None

def select_valid_paths(paths, num_workers=8):
    """
    Returns all paths that exist in the filesystem using multiple workers using imap
    
    """
    with Pool(num_workers) as pool:
        valid_paths = list(
            tqdm(
                pool.imap(exists, paths, chunksize=min(100, len(paths)//num_workers)),
                total=len(paths),
                desc="Selecting valid paths",
            )
        )
    return valid_paths


img_paths = select_valid_paths(img_paths_bin[:int(args.num_images*1.1)])
# Contains [path1, path2, ..., None, path_i, None ...] -> None values where the path is invalid


valid_mask = [path is not None for path in img_paths]
print(f"success_rate: {sum(valid_mask)}/{len(valid_mask)}")

if sum(valid_mask) < args.num_images:
    print(f"sum(valid_mask):{sum(valid_mask)} < args.num_images:{args.num_images}")

valid_img_paths = []
valid_labels = []
valid_times = []
valid_userIDs = []

for i, is_valid in enumerate(valid_mask):
    if not is_valid:
        continue
    valid_img_paths.append(img_paths[i])
    valid_labels.append(labels[i])
    valid_times.append(times[i])
    valid_userIDs.append(user_IDs[i])



reduced_img_paths = valid_img_paths[:args.num_images]
reduced_labels = valid_labels[:args.num_images]
reduced_times = valid_times[:args.num_images]
reduced_userIDs = valid_userIDs[:args.num_images]


print("Writing paths to file...")
with open(reduced_img_paths_fname, "w") as f:
    f.write("\n".join(reduced_img_paths))


# creating a torch binary file with the same format as the original
print("Creating torch binaries with reduced number of training samples...")
torch.save(reduced_img_paths, reduced_img_paths_bin_fname)
torch.save(reduced_labels, reduced_labels_fname)
torch.save(reduced_times, reduced_time_fname)
torch.save(reduced_userIDs, reduced_user_fname)
