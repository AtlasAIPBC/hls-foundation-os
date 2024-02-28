import cv2
import numpy as np
import os
import rasterio as rio

# generated_path = "/home/ada/prithvi/datasets/crop_classification/output/"
generated_path = "/home/ada/prithvi/datasets/test/crop_classification/output/"

mask_path = "/home/ada/prithvi/datasets/crop_classification/validation_chips/"

images = os.listdir(generated_path)
image_paths = [os.path.join(generated_path, f) for f in images]
mask_paths = [os.path.join(mask_path, f.replace("_merged_pred", ".mask")) for f in images]

images = [rio.open(path).read(1) for path in image_paths]
masks = [rio.open(path).read(1) for path in mask_paths]
print(len(images))
print(len(masks))

print(images[0].shape)
print(masks[0].shape)

def calculate_iou_multiple(gt_mask, pred_mask, class_values):

    # Ensure consistent shapes
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Ground truth and predicted masks must have the same shape.")

    ious = []
    for class_value in class_values:
        # Create class-specific masks
        class_gt_mask = gt_mask == class_value
        class_pred_mask = pred_mask == class_value

        # Calculate intersection and union
        intersection = np.logical_and(class_gt_mask, class_pred_mask).sum()
        union = np.logical_or(class_gt_mask, class_pred_mask).sum()

        # Calculate and append IoU with epsilon to avoid division by zero
        iou = intersection / (union + 1e-6)
        ious.append(iou)

    return ious

def calculate_accuracy(gt_mask, pred_mask):

    predicted_array = pred_mask
    ground_truth_array = gt_mask
    # Count number of correctly classified pixels
    correctly_classified_pixels = np.sum(predicted_array == ground_truth_array)
    # Calculate total number of pixels
    total_pixels = predicted_array.size
    # Calculate accuracy
    accuracy = (correctly_classified_pixels / total_pixels) * 100.0

    return accuracy

accuracies = [calculate_accuracy(mask, image) for mask, image in zip(masks, images)]

# Define class labels and values
# class_labels = ["No Data", "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
#                 "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa", 
#                 "Fallow/Idle Cropland", "Cotton", "Sorghum", "Other"]

# class_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

class_labels = ["Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
                "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa", 
                "Fallow/Idle Cropland", "Cotton", "Sorghum", "Other"]

class_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

class_ious = []
for mask, image in zip(masks, images):
    ious = calculate_iou_multiple(mask, image, class_values)
    class_ious.append(ious)

# Calculate mean IoU for each class
mean_class_ious = np.mean(np.array(class_ious), axis=0)

# Print results
print("Mean IoU per Class:")
for class_label, iou in zip(class_labels, mean_class_ious):
    print(f"{class_label}: {iou:.4f}")

print("Overall Mean IoU:", np.mean(mean_class_ious))
print("Mean Accuracy:", np.mean(accuracies))
