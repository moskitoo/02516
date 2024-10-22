import os
import argparse


def fix_naming(dataset_path: str):

    old_mask_dir_path = os.path.join(dataset_path, 'training/mask')
    new_mask_dir_path = os.path.join(dataset_path, 'training/1st_manual')

    change_dir_name(old_mask_dir_path, os.path.join(
        dataset_path, 'training', 'roi'))
    change_dir_name(new_mask_dir_path, os.path.join(
        dataset_path, 'training', 'masks'))


def change_dir_name(old_dir_path: str, new_dir_name: str):

    try:
        if os.path.exists(old_dir_path):
            os.rename(old_dir_path, new_dir_name)
            print(f"Directory renamed from '{
                  old_dir_path}' to '{new_dir_name}'")
        else:
            print(f"Directory '{old_dir_path}' does not exist.")
    except Exception as e:
        print(f"Error renaming directory: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rename "1st_manual" to "mask" in training directory to match the naming convention and change the original mask directory to a different name to avoid duplication - "roi".')
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset directory')

    args = parser.parse_args()

    fix_naming(args.dataset_path)
