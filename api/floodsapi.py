from gradio_client import Client

client = Client("https://ibm-nasa-geospatial-prithvi-100m-sen1floods11-demo.hf.space/--replicas/c0h6r/")
image_path = "/home/ada/prithvi/datasets/flood_mapping/v1.1/data/flood_events/HandLabeled/S2Hand/Somalia_886726_S2Hand.tif"
result = client.predict(
                image_path,
                fn_index=0
)
print(result)
