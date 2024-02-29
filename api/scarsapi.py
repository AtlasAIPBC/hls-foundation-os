from gradio_client import Client

client = Client(
    "https://ibm-nasa-geospatial-prithvi-100m-burn-scars-demo.hf.space/--replicas/iyigd/",
)
image_path = "/home/ada/prithvi/datasets/burn_scars/validation/subsetted_512x512_HLS.S30.T14SMC.2018213.v1.4_merged.tif"
result = client.predict(
        image_path,
        api_name="/partial"
)
print(result)
