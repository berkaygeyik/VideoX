import torch
from image_reader import get_data, print_data
from lib.train.data.bounding_box_utils import masks_to_bboxes

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


def create_bounding_boxes(segmentation_array, type):
    N, H, W = segmentation_array.shape

    all_bounding_boxes = []
    for i in range(N):
        segmentation_mask = segmentation_array[i, :, :]
        bounding_box = create_bounding_box_from_segmentation(segmentation_mask)
        all_bounding_boxes.append(bounding_box)

    file_name = "bounding_boxes_vessel.txt"
    img_index = 0 # index 206, 207th image
    with open(file_name, "w") as f:
        for i, bbox in enumerate(all_bounding_boxes):
            bb_str = " ".join(map(str, bbox))
            f.write(f"{bb_str}\n")
                
            if(i == img_index and type == "v"):
                print_data(i, bbox)

def create_zeros(segmentation_array):
    H, W, _, N = segmentation_array.shape
    with open("zeros.txt", "w") as file:
        for i in range(N):
            file.write("0")
            if i != N-1:
                file.write(",")

# Example usage
if __name__ == "__main__":
    segmentation_array = get_data()

    create_bounding_boxes(segmentation_array, 'v')
    create_zeros(segmentation_array)