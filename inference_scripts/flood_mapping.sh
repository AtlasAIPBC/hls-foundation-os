python3 model_inference.py \
    -config /home/ada/prithvi/hls-foundation-os/configs/sen1floods11_config.py \
    -ckpt /home/ada/prithvi/datasets/flood_mapping/weights/sen1floods11_Prithvi_100M.pth \
    -input /home/ada/prithvi/datasets/flood_mapping/v1.1/data/flood_events/HandLabeled/S2Hand/ \
    -output /home/ada/prithvi/datasets/flood_mapping/output/ \
    -input_type tif \
    -bands 0 1 2 3 4 5 \
