from gradio_client import Client
client = Client("https://ibm-nasa-geospatial-prithvi-100m-multi-temporal-f3e6c26.hf.space/--replicas/toj5n/")
image_path = "/home/ada/prithvi/datasets/crop_classification/validation_chips/chip_200_016_merged.tif"
result = client.predict(
                image_path,
                fn_index=0
)
print(result)
