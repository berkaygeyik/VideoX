import torch
from image_reader import get_data, print_data
from lib.train.data.bounding_box_utils import masks_to_bboxes
import os
import glob
import re


cur_path = os.path.dirname(__file__)

# Create bounding boxes from segmentation masks
def create_bounding_box_from_segmentation(segmentation_image):
    """
    Create a bounding box from a segmentation mask.
    :param segmentation_array: numpy array of the segmentation mask, shape = (H, W)
    :return: tensor containing the bounding box, shape = (4,)
    """
    segmentation_tensor = torch.tensor(segmentation_image, dtype=torch.float32)
    bounding_box = masks_to_bboxes(segmentation_tensor, fmt='t')
    return bounding_box.numpy()

# def create_bounding_boxes_from_segmentation(segmentation_image):

#     segmentation_tensor = torch.tensor(segmentation_image, dtype=torch.float32)
#     bounding_boxes = masks_to_bboxes_multi2(segmentation_tensor.numpy())
#     return bounding_boxes


def expand_bounding_box(bbox, expand_factor=0.1):

    x_center, y_center, width, height = bbox
    expanded_width = width * (1 + expand_factor)
    expanded_height = height * (1 + expand_factor)
    
    return [x_center, y_center, round(expanded_width, 1), round(expanded_height, 1)]


def create_bounding_boxes(segmentation_array, type, folder):
    N, H, W = segmentation_array.shape

    all_bounding_boxes = []
    for i in range(N):
        segmentation_mask = segmentation_array[i, :, :]
        bounding_box = create_bounding_box_from_segmentation(segmentation_mask)
        expanded_bounding_box = expand_bounding_box(bounding_box, expand_factor=0.1)
        all_bounding_boxes.append(expanded_bounding_box)

    new_path = os.path.relpath('../data/carotidartery', cur_path)
    file_name = "bounding_boxes_vessel.txt"
    img_index = 0 # index 206, 207th image
    with open(folder + "/" + file_name, "w") as f:
        for i, bbox in enumerate(all_bounding_boxes):
            bb_str = " ".join(map(str, bbox))
            f.write(f"{bb_str}\n")
                
            # if(i == img_index and type == "v"):
            #     print_data(i, bbox)

def create_zeros(segmentation_array, folder):
    N, H, W = segmentation_array.shape
    file_name = "zeros.txt"
    with open(folder + "/" + file_name, "w") as file:
        for i in range(N):
            file.write("0")
            if i != N-1:
                file.write(",")


def extract_number(folder_name):
    match = re.search(r'carotid-(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')


# Example usage
if __name__ == "__main__":
    main_directory = '/home/linux/VideoX/SeqTrack/data/carotidartery/'
    carotid_folders = sorted(glob.glob(os.path.join(main_directory, 'carotid-*')), key=extract_number)
    print(carotid_folders)

    for i, folder in enumerate(carotid_folders):
        print(i)
        print(folder)

    for i, folder in enumerate(carotid_folders):
        segmentation_array = get_data(i+1)
        create_bounding_boxes(segmentation_array, 'v', folder)
        create_zeros(segmentation_array, folder)
    