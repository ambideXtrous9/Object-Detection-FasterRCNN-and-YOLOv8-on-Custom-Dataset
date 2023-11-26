import os
from PIL import Image

def is_valid_image(img_path):
    try:
        Image.open(img_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

def remove_broken_images(folder_path, annotation_file_path):
    # Read the annotation file into a list
    with open(annotation_file_path, 'r') as file:
        annotations = file.readlines()

    # Filter out broken images
    valid_annotations = []
    for annotation in annotations:
        parts = annotation.split()
        img_name = parts[0]
        img_path = os.path.join(folder_path, img_name)
        if is_valid_image(img_path):
            valid_annotations.append(annotation)

    # Update the annotation file
    with open(annotation_file_path, 'w') as file:
        file.writelines(valid_annotations)

    # Remove broken images
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            if not is_valid_image(img_path):
                print(img_path)
                os.remove(img_path)

# Example usage
folder_path = "flickr_logos_dataset/flickr_logos_27_dataset_images"
annotation_file_path = "flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
remove_broken_images(folder_path, annotation_file_path)
