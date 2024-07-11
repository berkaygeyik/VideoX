import os
from PIL import Image

# input_folder = '/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/img_png'
# output_folder = '/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/img'

# Folder containing PNG files
input_folder = '/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/label_vessel'
# Folder where JPG files will be saved
output_folder = '/home/linux/VideoX/SeqTrack/data/carotidartery/carotid-1/label'

# If the output folder does not exist, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in folder
# Giriş klasöründeki tüm .png dosyalarını al
png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Dosyaları sırayla oku ve yeniden adlandırarak kaydet
for i, filename in enumerate(sorted(png_files), start=1):
    # Tam dosya yolunu oluşturun
    png_file = os.path.join(input_folder, filename)
    
    # Yeni JPG dosya adını ve yolunu oluşturun
    new_filename = f"{i:08d}.jpg"
    jpg_file = os.path.join(output_folder, new_filename)
        # Open image and save in JPG format
    with Image.open(png_file) as img:
        img = img.convert('RGB')
        img.save(jpg_file, 'JPEG')