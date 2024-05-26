import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


label_path = 'C:\\Users\\berka\\OneDrive\\VideoX\\SeqTrack\\data\\carotidartery\\label\\label.nii'
label_vessel_path = 'C:\\Users\\berka\\OneDrive\\VideoX\\SeqTrack\\data\\carotidartery\\label_vessel\\label_vessel.nii'

# label_img = nib.load(label_vessel) # for vessel, use this one
label_img = nib.load(label_path) # for plaque, use this one
label_data = label_img.get_fdata()

label_img_vessel = nib.load(label_vessel_path)
label_data_vessel = label_img_vessel.get_fdata()

# NIfTI dosyasını yükle
def print_data():
    # Print the shape of data
    print("Shape of data: ", label_data.shape)

    # get 200th element
    slice_200 = label_data[:, :, 0, 200]

    slice_200_rotated = np.rot90(slice_200, k=1) # rotate 90 degree
    slice_200_modified = np.flipud(slice_200_rotated) # symmetric horizontially

    # View Slice
    plt.imshow(slice_200_modified, cmap='gray')
    plt.title("modified image")
    plt.axis('off')
    plt.show()


def get_data():
    return label_data

def get_data_vessel():
    return label_data_vessel