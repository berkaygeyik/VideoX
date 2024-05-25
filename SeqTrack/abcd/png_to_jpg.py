import os
from PIL import Image

# Folder containing PNG files
input_folder = 'C:\\Users\\berka\\OneDrive\\VideoX\\SeqTrack\\data\\carotidartery\\img_png'
# Folder where JPG files will be saved
output_folder = 'C:\\Users\\berka\\OneDrive\\VideoX\\SeqTrack\\data\\carotidartery\\img'

# If the output folder does not exist, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # Full file path
        png_file = os.path.join(input_folder, filename)
        # New JPG file name and path
        jpg_file = os.path.join(output_folder, filename.replace('.png', '.jpg'))
        # Open image and save in JPG format
        with Image.open(png_file) as img:
            img = img.convert('RGB')
            img.save(jpg_file, 'JPEG')