import os
import argparse
import glob
import shutil


def copy_files(file_paths, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    for file_path in file_paths:
        
        file_path = glob.glob(os.path.join(file_path, '*'))[0]
        # file_path = glob.glob(os.path.join(file_path, '/*'))[0]
        
        filename = os.path.basename(file_path)
        destination_path = os.path.join(destination_dir, filename)
        
        try:
            shutil.copy(file_path, destination_path)
            print(f"Copied: {file_path} to {destination_path}")
        except Exception as e:
            print(f"Error copying {file_path}: {e}")

def restructure_dataset(dataset_path: str):
    
    image_paths = []
    label_paths = []

    object_dirs = glob.glob(os.path.join(dataset_path, '*/*'))
    
    images_destination_path = os.path.join(dataset_path, 'images')
    labels_destination_path = os.path.join(dataset_path, 'masks')

    for object_dir in object_dirs:
        if object_dir.endswith('_Image'):
            image_paths.append(object_dir)
        elif object_dir.endswith('_lesion'):
            label_paths.append(object_dir)

    copy_files(image_paths, images_destination_path)
    copy_files(label_paths, labels_destination_path)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Rename "1st_manual" to "mask" in training directory to match the naming convention and change the original mask directory to a different name to avoid duplication - "roi".')
#     parser.add_argument('dataset_path', type=str,
#                         help='Path to the dataset directory')

#     args = parser.parse_args()

#     restructure_dataset(args.dataset_path)

restructure_dataset('./PH2_Dataset_images/')
