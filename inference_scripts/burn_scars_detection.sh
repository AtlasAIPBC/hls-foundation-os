python3 model_inference.py \
    -config /home/ada/prithvi/hls-foundation-os/configs/burn_scars.py \
    -ckpt /home/ada/prithvi/datasets/burn_scars/weights/burn_scars_Prithvi_100M.pth \
    -input /home/ada/prithvi/datasets/burn_scars/validation \
    -output /home/ada/prithvi/datasets/burn_scars/output/ \
    -input_type tif \
    -bands 0 1 2 3 4 5 \
