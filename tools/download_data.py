import os
import argparse
import requests

def download_file(url, save_path):
    r = requests.get(url, allow_redirects=True)
    with open(save_path, 'wb') as f:
        f.write(r.content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_data', default='fashion', type=str)
    parser.add_argument('--save_dir', default='./raw_data', type=str)
    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.name_data == 'fashion':
        # URLs of the train and test list files
        train_list_url = "https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_train.txt"
        test_list_url = "https://vision.cs.ubc.ca/datasets/fashion/resources/fashion_dataset/fashion_test.txt"
        
        # Download the train and test list files
        download_file(train_list_url, os.path.join(save_dir, "fashion_train.txt"))
        download_file(test_list_url, os.path.join(save_dir, "fashion_test.txt"))

        # Read the URLs from the files
        with open(os.path.join(save_dir, 'fashion_train.txt'), "r") as f_train:
            train_files = f_train.readlines()
        
        with open(os.path.join(save_dir, 'fashion_test.txt'), "r") as f_test:
            test_files = f_test.readlines()

        # Create directories for train and test data
        os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)

        # Download training data
        for video_url in train_files:
            video_url = video_url.strip()
            file_name = video_url.split("/")[-1]
            download_file(video_url, os.path.join(save_dir, "train", file_name))

        # Download test data
        for video_url in test_files:
            video_url = video_url.strip()
            file_name = video_url.split("/")[-1]
            download_file(video_url, os.path.join(save_dir, "test", file_name))
    print(f"Downloaded raw video to {save_dir}")