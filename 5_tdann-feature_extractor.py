import os
from os.path import join as pjoin
basedir = r'/n02dat01/users/qdzhao/THINGS'
code_path = pjoin(basedir, r'property_analysis/code/cnn-som/TDANN/TDANN-main')
positions_dir = pjoin(basedir, r'property_analysis/code/cnn-som/TDANN', r'tdann_data/tdann/positions/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019_NBVER2')
weights_path = pjoin(basedir, r'property_analysis/code/cnn-som/TDANN', r'tdann_data/tdann/checkpoints', f'model_final_checkpoint_phase199.torch')
image_path = pjoin(basedir, r'property_analysis/code/cnn-som/TDANN', r'tdann_data/image', f'test_img.jpg')

import sys
sys.path.append(code_path)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from src.positions import NetworkPositions
from src.model import load_model_from_checkpoint, LAYERS
from src.data import load_image, create_dataloader
from src.features import FeatureExtractor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

# load positions from each model layer
network_positions = NetworkPositions.load_from_dir(positions_dir)

# load model from checkpoint and send to appropriate device
model = load_model_from_checkpoint(weights_path)
model = model.to(DEVICE)


img_set_dir = pjoin(r'/n02dat01/users/qdzhao/THINGS/THINGS_stimuli/THINGS/Images')
img_partitions = os.listdir(img_set_dir)

# Create the saving directory if not existing
save_dir = pjoin(r'/n02dat01/users/qdzhao/THINGS/THINGS_stimuli/THINGS/derivatives/dnn_feature_maps/TDANN')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# image processing
centre_crop = trn.Compose([
    trn.Resize((224,224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

feat_list = ['layer1.0','layer1.1','layer2.0','layer2.1','layer3.0','layer3.1','layer4.0','layer4.1']

# ['object_images_D-K', 'object_images_L-Q', 'password.txt', 'object_images_T-Z', 'object_images_A-C', 'object_images_R-S']


for p in img_partitions:
    if p=='object_images_D-K' or p=='object_images_L-Q' or p=='password.txt':
        # break # 跳出本层循环
        continue # 跳出本次循环
    part_dir = pjoin(img_set_dir, p) # imageset A-Z
    img_partitions_ = os.listdir(part_dir)
    for pp in img_partitions_:
        part_dir_ = pjoin(part_dir, pp) # image category aardvark to zucchini
        image_list = []
        for root, dirs, files in os.walk(part_dir_):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".JPEG"):
                    image_list.append(pjoin(root,file))
        image_list.sort()
        # Extract and save the feature maps
        feats_ = {}
        for i, image in enumerate(image_list):
            # img = Image.open(image).convert('RGB')
            # input_img = V(centre_crop(img).unsqueeze(0))
            # if torch.cuda.is_available():
            #     input_img=input_img.cuda()
            # create a dataloader to serve the single image we're pointing to
            dataloader = create_dataloader(load_image(image))
            # extract features for all layers, also storing the images and labels
            extractor = FeatureExtractor(dataloader, n_batches=1, verbose=True)
            features, inputs, labels = extractor.extract_features(model, LAYERS, return_inputs_and_labels=True)

            feats = {}
            for ly in feat_list:
                feats[ly] = features[ly].flatten()
            feats_[i] = feats
        feat_avg = {}
        for ly in feat_list:
            tmp = []
            for ii, image in enumerate(image_list):
                tmp.append(feats_[ii][ly].squeeze())
            tmp_ = np.array(tmp)
            tmp_avg = np.mean(tmp_,0)
            feat_avg[ly] = tmp_avg
        
        np.save(os.path.join(save_dir, pp), feat_avg)



# # view the input image
# # remove batch dim, and move channels to the end
# raw = inputs[0].squeeze().transpose(1, 2, 0)

# # normalize image to the range [0, 1]
# normed = (raw - np.min(raw)) / np.ptp(raw)

# # plot
# fig, ax = plt.subplots(figsize=(1, 1))
# ax.imshow(normed)
# ax.axis("off")

# # show response magnitude in each layer
# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=4,
#     figsize=(16, 8)
# )

# for ax, layer in zip(axes.ravel(), LAYERS):
#     # coordinates is an N x 2 matrix of unit positions, in mm
#     coordinates = network_positions.layer_positions[layer].coordinates
    
#     # responses is an N-dimensional response vector from each layer
#     responses = features[layer]
    
#     # plot points, scaling by total size of the tissue in each layer
#     extent = np.ptp(coordinates)
#     ax.scatter(*coordinates.T, c=responses, cmap='magma', s=extent / 100)
#     ax.set_title(layer)