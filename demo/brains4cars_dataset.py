
import os
from os.path import join
import sys
from torch.utils.data import Dataset

import cv2
import glob
import time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("SCRIPT ", SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
rng = np.random.default_rng(seed=42)

class B4CDataset(Dataset):
    """
    Assumes that B4C has no labels
    """
    def __init__(self, class_names, root_path, camera, split, create_dataset=False):
        super().__init__()

        self.camera = camera
        self.split = split
        self.root_path = root_path
        self.class_names = class_names

        self.root_split_path = self.root_path / self.camera

        outrootdir = self.root_path / str(self.camera+"_processed")

        if create_dataset:
            self.create_dataset(outrootdir)
            self.generate_imagesets(outrootdir)
        
        imageset_split_path = join(outrootdir, "ImageSets", f'{self.split}.txt')
        self.img_list = [line.strip() for line in open(imageset_split_path, 'r')]

    @staticmethod
    def avi_to_jpg(input_dir, output_dir):
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get a list of all files in the input directory
        files = [file for file in os.listdir(input_dir) if file.endswith('.avi')]

        # Loop through all files
        for file in files:
            start_time = time.time()
            # Check if the file is an AVI video
            if file.lower().endswith(".avi"):
                avi_file = os.path.join(input_dir, file)

                # Open the video file
                cap = cv2.VideoCapture(avi_file)

                # Check if the video file was opened successfully
                if not cap.isOpened():
                    print(f"Error opening video file: {avi_file}")
                    continue

                # Get the frames per second (fps) and total number of frames
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                avi_name = avi_file.split('.')[0]
                # Loop through the frames and save each frame as a JPEG image
                for frame_num in range(total_frames):
                    # Read the frame
                    ret, frame = cap.read()

                    # Check if the frame was read successfully
                    if not ret:
                        print(f"Error reading frame {frame_num} in: {avi_file}")
                        break

                    # Generate the output file name (e.g., frame_001.jpg)
                    filename = file.split('.')[0]
                    output_video_dir = join(output_dir, filename)
                    if not os.path.exists(output_video_dir):
                        os.mkdir(output_video_dir)
                    output_file = join(output_video_dir, f"{str(frame_num).zfill(4)}.jpg")
                    
                    # Save the frame as a JPEG image
                    cv2.imwrite(output_file, frame)

                # Release the video capture object
                cap.release()
            print(f'Took {time.time()-start_time} seconds to process video {file}')

        print("Conversion completed successfully.")

    def create_dataset(self, outrootdir):
        actions_subdirs = sorted(glob.glob(os.path.join(self.root_split_path, "*")))
        actions_subdir_names = [subdir.split('/')[-1] for subdir in actions_subdirs]

        print("Starting to create dataset")
        for subdir_idx, subdir_name in enumerate(actions_subdir_names):
            outdir = join(outrootdir, subdir_name)

            if not os.path.exists(outdir):
                print(f'Outdir {outdir} does not exist, creating now...')
                os.makedirs(outdir)
            
            indir = actions_subdirs[subdir_idx]
            self.avi_to_jpg(indir, outdir)
        print("Finished creating dataset...")

    def get_all_file_paths(self, directory_path):
        """
        Recursively get the file paths of all the files in the specified directory.

        Parameters:
            directory_path : str
                The path to the directory to be traversed.

        Returns:
            list
                A list containing all the file paths.
        """
        file_paths = []

        for entry in os.scandir(directory_path):
            if entry.is_file():
                file_paths.append(entry.path)
            elif entry.is_dir():
                file_paths.extend(self.get_all_file_paths(entry.path))

        return file_paths

    @staticmethod
    def write_list_to_file(file_path, data_list):
        """
        Create/overwrite a .txt file and write each line of the Python list to a new line in the file.

        Parameters:
            file_path : str
                The path to the .txt file.
            data_list : list
                The Python list containing data to write to the file.
        """
        with open(file_path, 'w') as file:
            file.writelines(f"{item}\n" for item in data_list)

    def generate_imagesets(self, outrootdir):
        actions_subdirs = sorted(glob.glob(os.path.join(outrootdir, "*")))
        actions_subdirs = [valid_subdir for valid_subdir in actions_subdirs if "ImageSets" not in valid_subdir]
        actions_subdir_names = [subdir.split('/')[-1] for subdir in actions_subdirs]
        
        print("Generating imagesets...")
        # Designate split as 0.7, 0.15, 0.15
        split_dict = {"train": [], "val": [], "test": []}
        for subdir in actions_subdirs:
            # Get all file_paths
            img_list = self.get_all_file_paths(subdir)

            indices = np.arange(0, len(img_list), 1)
            rng.shuffle(indices)

            dataset_size = len(img_list)
            train_len   = int(0.7*dataset_size)
            val_len     = int(train_len + 0.15*dataset_size)
            test_len    = dataset_size

            train, val, test    = indices[:train_len], indices[train_len:val_len], \
                indices[val_len:test_len]

            split_dict["train"].extend(np.array(img_list)[train].tolist())
            split_dict["val"].extend(np.array(img_list)[val].tolist())
            split_dict["test"].extend(np.array(img_list)[test].tolist())

        # Write images sets to .txt files
        imagesets_dir = join(outrootdir, "ImageSets")
        if not os.path.exists(imagesets_dir):
            os.mkdir(imagesets_dir)
        for split_key, split_list in split_dict.items():
            split_path = join(imagesets_dir, f'{split_key}.txt')
            print(f'Saving imageset file for split {split_path}')
            self.write_list_to_file(split_path, split_list)

    def __len__(self):
        return len(self.img_list)

    def get_image(self, index):
        img_file = self.img_list[index]
        assert os.path.exists(img_file)
        image_bgr = cv2.imread(img_file)  # Read the image in BGR format
        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        return image_bgr, img_file

    def __getitem__(self, index):
        img, img_file   = self.get_image(index)
        return img, img_file

    def collate_fn(self, batch):
        img_batch = [bi[0] for bi in batch]
        img_file_batch = [bi[1] for bi in batch]

        return img_batch, img_file_batch


