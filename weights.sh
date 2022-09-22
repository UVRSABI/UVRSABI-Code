#!/bin/bash

cd utils/LEDNet
mkdir -p save/logs
cd ./save/logs
# Check if the model weights have been downloaded before
if test -f "model_best.pth"; then
    echo "Roof model weights exist"
else
    echo "Roof model weights do not exist"
    echo "Downloading model weights"
    gdown 1E-FLi3byKxCcfKAGfqWAkp7Li4poKdKO
    echo "Done!"
fi
cd ../../../..
echo $(pwd)
cd RoofLayoutEstimation/Detic

if test -f "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"; then
    echo "NSE model weights exist"
else
    echo "NSE model weights do not exist"
    echo "Downloading model weights"
    gdown 1RS2a1V-Gm2c4j7PbR9qg_xdzI1ohj6lz
    echo "Done!"
fi
cd ../..