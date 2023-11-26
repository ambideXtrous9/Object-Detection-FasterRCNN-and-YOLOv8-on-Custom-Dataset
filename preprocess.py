import os
from PIL import Image

def is_valid_image(img_path):
    try:
        Image.open(img_path).verify()
        return True
    except (IOError, SyntaxError):
        return False

def remove_broken_and_invalid_entries(folder_path, annotation_file_path):
    
    total_images_before = len(os.listdir(folder_path))
    
    with open(annotation_file_path, 'r') as file:
        total_entries_before = len(file.readlines())

    # Read the annotation file into a list
    with open(annotation_file_path, 'r') as file:
        annotations = file.readlines()

    # Filter out broken and invalid entries
    valid_annotations = []
    for annotation in annotations:
        parts = annotation.split()
        img_name, class_name, _, xmin, ymin, xmax, ymax = parts

        # Check if image is valid
        img_path = os.path.join(folder_path, img_name)
        if not is_valid_image(img_path):
            continue

        # Check if bounding box is valid
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        if xmin >= xmax or ymin >= ymax:
            continue

        # If both checks pass, add the annotation to the valid list
        valid_annotations.append(annotation)

    # Update the annotation file
    with open(annotation_file_path, 'w') as file:
        file.writelines(valid_annotations)

    # Remove broken and invalid images
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            if not is_valid_image(img_path):
                os.remove(img_path)
            else:
                img_name = file
                annotation_exists = any(img_name in annotation for annotation in valid_annotations)
                if not annotation_exists:
                    print(img_path)
                    os.remove(img_path)

    # Count the total number of images after removal
    total_images_after = len(os.listdir(folder_path))
    
    total_entries_after = len(valid_annotations)

    # Print the results
    print(f"Total number of entries before: {total_entries_before}")
    print(f"Total number of entries after Removal: {total_entries_after}")

    # Print the results
    print(f"Total number of images before Removal: {total_images_before}")
    print(f"Total number of images after Removal: {total_images_after}")

# Example usage
folder_path = "flickr_logos_dataset/flickr_logos_27_dataset_images"
annotation_file_path = "flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
remove_broken_and_invalid_entries(folder_path, annotation_file_path)

