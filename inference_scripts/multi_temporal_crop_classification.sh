python3 model_inference.py \
    -config /home/ada/prithvi/hls-foundation-os/configs/multi_temporal_crop_classification.py \
    -ckpt /home/ada/prithvi/datasets/crop_classification/weights/multi_temporal_crop_classification_Prithvi_100M.pth \
    -input /home/ada/prithvi/datasets/crop_classification/validation_chips/ \
    -output /home/ada/prithvi/datasets/crop_classification/output/ \
    -input_type tif \
    -bands 0 1 2 3 4 5 \
