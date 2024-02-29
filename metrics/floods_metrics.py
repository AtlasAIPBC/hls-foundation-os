import cv2
import numpy as np
import os
import rasterio as rio

generated_path = "/home/ada/prithvi/datasets/test/flood_mapping/output/"
mask_path = "/home/ada/prithvi/datasets/flood_mapping/v1.1/data/flood_events/HandLabeled/LabelHand/"

images = os.listdir(generated_path)
image_paths = [os.path.join(generated_path, f) for f in images]
mask_paths = [os.path.join(mask_path, f.replace("S2Hand_pred", "LabelHand")) for f in images]

images = [rio.open(path).read(1) for path in image_paths]
masks = [rio.open(path).read(1) for path in mask_paths]

print(images[0].shape)
print(masks[0].shape)

def calculate_iou_separate(gt_mask, pred_mask):
    """
    Calculates the Intersection over Union (IoU) score for flood and non-flood classes.

    Returns:
        tuple: A tuple containing two floats:
               - First element: IoU score for the flood class (pixel value = 1).
               - Second element: IoU score for the non-flood class (pixel values = 0 and 2).
    """

    # Ensure consistent shapes
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("Ground truth and predicted masks must have the same shape.")

    # Define masks for each class
    flood_gt_mask = gt_mask == 1
    non_flood_gt_mask = gt_mask == 0
    # non_flood_gt_mask = np.logical_or(gt_mask == 0, gt_mask == 2)

    flood_pred_mask = pred_mask == 1
    non_flood_pred_mask = pred_mask == 0
    # non_flood_pred_mask = np.logical_or(pred_mask == 0, pred_mask == 2)

    # Calculate intersection and union for each class
    flood_intersection = np.logical_and(flood_gt_mask, flood_pred_mask).sum()
    flood_union = np.logical_or(flood_gt_mask, flood_pred_mask).sum()

    non_flood_intersection = np.logical_and(non_flood_gt_mask, non_flood_pred_mask).sum()
    non_flood_union = np.logical_or(non_flood_gt_mask, non_flood_pred_mask).sum()

    # Calculate and return IoU scores with epsilon to avoid division by zero
    flood_iou = flood_intersection / (flood_union + 1e-6)
    non_flood_iou = non_flood_intersection / (non_flood_union + 1e-6)

    return flood_iou, non_flood_iou

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
flood_ious = [calculate_iou_separate(mask, image)[0] for mask, image in zip(masks, images)]
non_flood_iou = [calculate_iou_separate(mask, image)[1] for mask, image in zip(masks, images)]

# ious = [calculate_iou(mask, image) for mask, image in zip(masks, images)]
print(len(flood_ious))
print(len(non_flood_iou))
print("Mean Flood IoU:", round(np.mean(flood_ious), 3))
print("Mean No Water/Flood IoU:", round(np.mean(non_flood_iou), 3))
print("Mean IoU", round(np.mean([np.mean(flood_ious), np.mean(non_flood_iou)]), 3))
print("Mean Accuracy:", round(np.mean(accuracies), 3))

# "Somalia_886726_LabelHand.tif"
# "Somalia_886726_S2Hand.tif"
# "Paraguay_80102_S2Hand_pred.tif"
