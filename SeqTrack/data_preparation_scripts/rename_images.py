import os
import glob

def rename_images(directory, new_start, new_extension):
    file_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_list.append(filename)

    file_list.sort()

    counter = new_start
    for old_name in file_list:
        new_name = f"{counter:08d}.{new_extension}"
        os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))
        counter += 1


# directory_images = "/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/img"
# directory_labels = "/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/label"

main_directory = '/home/linux/VideoX/SeqTrack/data/carotidartery/'

carotid_folders = sorted(glob.glob(os.path.join(main_directory, 'carotid-*')))


for folder in carotid_folders:
    img_path = os.path.join(folder, "img")
    label_path = os.path.join(folder, "label")
    new_start_number = 1

    rename_images(img_path, new_start_number, "jpg")
    rename_images(label_path, new_start_number, "png")

