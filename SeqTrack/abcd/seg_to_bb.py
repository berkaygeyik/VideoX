import numpy as np
import torch
from nifti_reader import get_data
from lib.train.data.bounding_box_utils import masks_to_bboxes, masks_to_bboxes_multi2
import cv2 as cv

# def masks_to_bboxes(mask, fmt='c'):
#     """
#     Convert a mask tensor to a bounding box.
#     :param mask: Tensor of masks, shape = (H, W)
#     :param fmt: bbox layout. 'c' => "center + size" or (x_center, y_center, width, height)
#                              't' => "top left + size" or (x_left, y_top, width, height)
#     :return: tensor containing a bounding box, shape = (4,)
#     """
#     mx = mask.sum(dim=-2).nonzero()
#     my = mask.sum(dim=-1).nonzero()
#     if len(mx) > 0 and len(my) > 0:
#         x_min, x_max = mx.min().item(), mx.max().item()
#         y_min, y_max = my.min().item(), my.max().item()
#         if fmt == 'c':
#             cx = (x_min + x_max) / 2
#             cy = (y_min + y_max) / 2
#             w = x_max - x_min + 1
#             h = y_max - y_min + 1
#             return torch.tensor([cx, cy, w, h], dtype=torch.float32)
#         elif fmt == 't':
#             w = x_max - x_min + 1
#             h = y_max - y_min + 1
#             return torch.tensor([x_min, y_min, w, h], dtype=torch.float32)
#         else:
#             raise ValueError("Undefined bounding box layout '%s'" % fmt)
#     else:
#         return torch.tensor([0, 0, 0, 0], dtype=torch.float32)


# Create bounding boxes from segmentation masks
def create_bounding_box_from_segmentation(segmentation_array):
    """
    Create a bounding box from a segmentation mask.
    :param segmentation_array: numpy array of the segmentation mask, shape = (H, W)
    :return: tensor containing the bounding box, shape = (4,)
    """
    segmentation_tensor = torch.tensor(segmentation_array, dtype=torch.float32)
    bounding_box = masks_to_bboxes(segmentation_tensor, fmt='t')
    return bounding_box

def create_bounding_boxes_from_segmentation(segmentation_array):

    segmentation_tensor = torch.tensor(segmentation_array, dtype=torch.float32)
    bounding_boxes = masks_to_bboxes_multi2(segmentation_tensor.numpy())
    return bounding_boxes


def create_bounding_boxes(segmentation_array):
    H, W, _, N = segmentation_array.shape
    all_bounding_boxes = []
    # for i in range(N):


# Example usage
if __name__ == "__main__":
    # Example input: numpy array of shape (477, 516)
    # segmentation_array = np.random.randint(0, 2, (477, 516))  # Example data
    # print(segmentation_array.shape)
    segmentation_array = get_data()
    # print(segmentation_array.shape)
    # print(segmentation_array[:, :, 0, 210].shape)
    
    segmentation_image = segmentation_array[:, :, 0, 210]
    print(segmentation_array.shape)
    
    
    bounding_box = create_bounding_box_from_segmentation(segmentation_image)
    print("Bounding Box:", bounding_box.numpy())
    # save_bounding_boxes_to_file(bounding_boxes, 'bounding_boxes.txt')