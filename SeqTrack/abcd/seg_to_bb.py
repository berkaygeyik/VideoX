import torch
from nifti_reader import get_data
from lib.train.data.bounding_box_utils import masks_to_bboxes, masks_to_bboxes_multi2

# Create bounding boxes from segmentation masks
def create_bounding_box_from_segmentation(segmentation_image):
    """
    Create a bounding box from a segmentation mask.
    :param segmentation_array: numpy array of the segmentation mask, shape = (H, W)
    :return: tensor containing the bounding box, shape = (4,)
    """
    segmentation_tensor = torch.tensor(segmentation_image, dtype=torch.float32)
    bounding_box = masks_to_bboxes(segmentation_tensor, fmt='t')
    return bounding_box

def create_bounding_boxes_from_segmentation(segmentation_image):

    segmentation_tensor = torch.tensor(segmentation_image, dtype=torch.float32)
    bounding_boxes = masks_to_bboxes_multi2(segmentation_tensor.numpy())
    return bounding_boxes


def create_bounding_boxes(segmentation_array):
    H, W, _, N = segmentation_array.shape
    all_bounding_boxes = []
    for i in range(N):
        segmentation_mask = segmentation_array[:, :, 0, i]
        bounding_boxes = create_bounding_boxes_from_segmentation(segmentation_mask)

        bounding_boxes_of_mask = []
        for bb in bounding_boxes:
            bounding_boxes_of_mask.append(bb.numpy())
        all_bounding_boxes.append(bounding_boxes_of_mask)

    with open("bounding_boxes.txt", "w") as f:
        for i, bboxes in enumerate(all_bounding_boxes):
            count=0
            for bb in bboxes:
                bb_str = " ".join(map(str, bb))
                f.write(f"{i+1}:{bb_str}\n")
                count+=1
            if count == 0: # remove here to remove empty images from output
                f.write(f"{i+1}:\n")

# Example usage
if __name__ == "__main__":
    segmentation_array = get_data()
    
    segmentation_image = segmentation_array[:, :, 0, 210]
    print(segmentation_array.shape)
    
    create_bounding_boxes(segmentation_array)