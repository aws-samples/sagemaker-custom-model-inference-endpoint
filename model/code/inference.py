import os
import torch
import numpy as np
import rasterio
import yaml
from prithvi.Prithvi import MaskedAutoencoderViT
import sys
import json
import base64
import io
#from PIL import Image
#import matplotlib.pyplot as plt

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

means = 0
stds = 0
#raster_data = 0
#Visualization
def enhance_raster_for_visualization(raster, ref_img=None):
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized_raster = (raster[channel] - mins) / (maxs - mins)
        normalized_raster[~valid_mask] = 0
        clipped = np.clip(normalized_raster, 0, 1)
        channels.append(clipped)
    clipped = np.stack(channels)
    channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
    rgb = channels_last[..., ::-1]
    return rgb
"""
def plot_image_mask_reconstruction(normalized, mask_img, pred_img):
    # Mix visible and predicted patches
    rec_img = normalized.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]  # binary mask: 0 is keep, 1 is remove

    mask_img_np = mask_img.numpy().reshape(6, 224, 224).transpose((1, 2, 0))[..., :3]

    rec_img_np = (rec_img.numpy().reshape(6, 224, 224) * stds) + means
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    for subplot in ax:
        subplot.axis('off')

    ax[0].imshow(enhance_raster_for_visualization(raster_data))
    masked_img_np = enhance_raster_for_visualization(raster_data).copy()
    masked_img_np[mask_img_np[..., 0] == 1] = 0
    ax[1].imshow(masked_img_np)
    ax[2].imshow(enhance_raster_for_visualization(rec_img_np, ref_img=raster_data))"""

def model_fn(model_dir):
    # implement custom code to load the model
    # load weights
#!cat ./prithvi/Prithvi_100M.pt
    weights_path = "./prithvi/Prithvi_100M.pt"
    checkpoint = torch.load(weights_path, map_location="cpu")

# read model config
    model_cfg_path = "./prithvi/Prithvi_100M_config.yaml"
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)

    model_args, train_args = model_config["model_args"], model_config["train_params"]

    global means
    global stds 
    means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
    stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)
# let us use only 1 frame for now (the model was trained on 3 frames)
    model_args["num_frames"] = 1

# instantiate model
    model = MaskedAutoencoderViT(**model_args)
    model.eval()

# load weights into model
# strict=false since we are loading with only 1 frame, but the warning is expected
    del checkpoint['pos_embed']
    del checkpoint['decoder_pos_embed']
    _ = model.load_state_dict(checkpoint, strict=False)
    
    return model

def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()

        # load first 6 bands
        img = img[:6]

        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img

def preprocess_image(image):
    # normalize image
    normalized = image.copy()
    normalized = ((image - means) / stds)
    normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized

def input_fn(input_data, content_type):
    # decode the input data  (e.g. JSON string -> dict)
    # statistics used to normalize images before passing to the model
    raster_path = "https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-demo/resolve/main/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"
    #global raster_data
    raster_data = load_raster(raster_path, crop=(224, 224))
    return raster_data

def predict_fn(data, model):
    normalized = preprocess_image(data)
    with torch.no_grad():
        mask_ratio = 0.5
        _, pred, mask = model(normalized, mask_ratio=mask_ratio)
        mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
        pred_img = model.unpatchify(pred).detach().cpu()#.numpy()
        rec_img = normalized.clone()
        rec_img[mask_img == 1] = pred_img[mask_img == 1]
        rec_img_np = (rec_img.numpy().reshape(6, 224, 224) * stds) + means
        print(rec_img_np.shape)
        #plot_image_mask_reconstruction(normalized, mask_img, pred_img)
        return enhance_raster_for_visualization(rec_img_np, ref_img=data)

def output_fn(prediction, accept):
    #print("Output:" + prediction)
    print(prediction.shape)
    return prediction.tobytes()



def tiff_to_json(response):
    try:

        pred_bytes = io.BytesIO()
        response.save(pred_bytes, format='TIFF')
        pred_bytes = pred_bytes.getvalue()

        # Convert bytes to base64 strings
        pred_base64 = base64.b64encode(pred_bytes).decode('utf-8')

        # Create JSON object
        json_data = {
            "pred_tiff_base64": pred_base64,
            "metadata": {
                "pred_shape": response.size
                # Add more metadata if needed
            }
        }

        return json_data
    except Exception as e:
        print("Error converting TIFF files to JSON:", e)
        return None

'''
def main():
    args = sys.argv

    payload = args[1]

    response = output_fn(
        predict_fn(
            input_fn(payload, "application/x-image"),
            model_fn("../")
    ),
    "image/tiff"
)
'''
    #tiff_bytes = response
    #tiff_image_np = np.frombuffer(tiff_bytes).reshape((224, 224, 3))
    
    #TESTING 
    #raster_data = input_fn(payload, "application/x-image")
    #raster_for_visualization = enhance_raster_for_visualization(raster_data)
    #plt.imshow(tiff_image_np)
   # plt.show()


if __name__ == "__main__":
    main()

#Gets the image right
#Show correct prediction image
#Sends data over JSON correctly?