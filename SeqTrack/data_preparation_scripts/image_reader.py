import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            
            images.append(img_array)
    return np.array(images)

folder_path = '/home/linux/VideoX/SeqTrack/data/carotidartery/'
label_data_sample = load_images_from_folder('/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-2/label')


def print_data(img_index, bbox = []):
    # Print the shape of data
    print("Shape of data: ", label_data_sample.shape)
    # get 200th element
    slice_207 = label_data_sample[img_index, :, :]
    slice_207_transposed = slice_207.astype(float)

    plt_title = f"modified image {img_index + 1}"

    fig, ax = plt.subplots()
    ax.imshow(slice_207_transposed, cmap='gray')
    plt.title(plt_title)
    plt.axis('off')
    print(bbox)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


def get_data(i):
    return load_images_from_folder(folder_path + "carotid-" + str(i) + "/label").astype(float)

# print(label_data.shape)
# print(label_data[207][200][300][0])

if __name__ == "__main__":
    print_data(125)