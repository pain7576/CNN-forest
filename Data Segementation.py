import os
import cv2
import xml.etree.ElementTree as ET

# Define paths
image_folder = r'D:\MLME project\Mlme\pythonProject\data\Forest2\train_images'
annotation_folder = r'D:\MLME project\Mlme\pythonProject\data\Forest2\train_annotations'
output_folder = r'D:\MLME project\Mlme\pythonProject\data\Forest2\pca_train_health'

# Desired dimensions for the resized images
desired_width = 100
desired_height = 100

# Create output folders if they don't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each XML file in the annotation folder
for xml_file in os.listdir(annotation_folder):
    if xml_file.endswith('.xml'):
        # Parse XML file
        xml_path = os.path.join(annotation_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image file name
        image_file = root.find('filename').text
        image_path = os.path.join(image_folder, image_file)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image file {image_file} not found, skipping.")
            continue

        # Read the image
        image = cv2.imread(image_path)

        # Iterate through each object in the annotation file
        object_count = 0  # To create unique filenames
        for obj in root.findall('object'):
            damage_tag = obj.find('damage').text

            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Crop the image
            cropped_image = image[ymin:ymax, xmin:xmax]

            # Resize the cropped image
            resized_image = cv2.resize(cropped_image, (desired_width, desired_height))

            # Create folder for the damage tag if it doesn't exist
            damage_folder = os.path.join(output_folder, damage_tag)
            if not os.path.exists(damage_folder):
                os.makedirs(damage_folder)

            # Save the resized image with a unique filename
            cropped_image_file = f'{os.path.splitext(image_file)[0]}_{damage_tag}_{object_count}.jpg'
            cropped_image_path = os.path.join(damage_folder, cropped_image_file)
            cv2.imwrite(cropped_image_path, resized_image)

            # Increment the object counter
            object_count += 1

print("Cropping, resizing, and segregation completed.")