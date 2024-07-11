import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


label_path = '/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/label/label.nii'
label_vessel_path = '/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/label_vessel/label_vessel.nii'

# label_img = nib.load(label_vessel) # for vessel, use this one
label_img = nib.load(label_path) # for plaque, use this one
label_data = label_img.get_fdata()

label_img_vessel = nib.load(label_vessel_path)
label_data_vessel = label_img_vessel.get_fdata()

# Load NIfTI file
def print_data(img_index, bboxes = []):
    # Print the shape of data
    print("Shape of data: ", label_data_vessel.shape)
    # get 200th element
    slice_207 = label_data_vessel[:, :, 0, img_index - 1]
    print(slice_207[300])

    plt_title = f"modified image {img_index + 1}"
    # View Slice
    
    fig, ax = plt.subplots()
    ax.imshow(slice_207, cmap='gray')
    plt.title(plt_title)
    plt.axis('off')

    if bboxes:
        for bbox in bboxes:
            rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.show()


def get_data():
    print(label_data.shape)
    return label_data

def get_data_vessel():
    return label_data_vessel

print(label_data_vessel.shape)
print(label_data_vessel[300][200][0][207])

if __name__ == "__main__":
    print_data(207)