#!/bin/bash
# conda init bash
# This script is used to estimate the roof layout of a building
cd RoofLayoutEstimation
results_path=../RoofLayoutEstimationResults
LOG=$results_path/log.txt
exec > >(tee $LOG)

echo "Extracting images from the video"
rm -rf images
python ../utils/VideoToImage.py --video $1 --savedir images
echo "Done!"
touch $results_path/out.log
echo "Estimating the roof masks"
cd LEDNet/test
chmod 777 test.py
# rm -rf RoofMasks
# python test.py --datadir ../../images --resultdir ../../RoofMasks >> ../../$results_path/out.log
cd ../../
echo "Done!"

echo "Displaying Roof Mask Results"
chmod 777 saveroofmaskresults.py
python saveroofmaskresults.py -i images -r RoofMasks -s $results_path/intermediate_results>> $results_path/out.log
echo "Done!"

echo "Estimating the NSE Masks"
# rm -rf ObjectMasks
# mkdir ObjectMasks
cd Detic
chmod 777 demo.py
# python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ../images/*.png --output ../ObjectMasks --vocabulary custom --custom_vocabulary solar_array,air_conditioner,vent,box,sink --confidence-threshold 0.5 --opts MODEL.WEIGHTS Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth >> ../$results_path/out.log
cd ..
echo "Done!"

echo "Displaying Intermediate Results before stitching"
chmod 777 generateintermediateresults.py
python generateintermediateresults.py -i images -r RoofMasks -o ObjectMasks -s $results_path/intermediate_results >> $results_path/out.log
echo "Done!"

echo "Stitching Images"
chmod 777 stitch.py
python stitch.py -i images -o ObjectMasks -r RoofMasks -s $results_path/intermediate_results >> $results_path/out.log
echo "Done!"

echo "Caclulating the percentage occupancy"
chmod 777 calculateoccupancy.py
python calculateoccupancy.py -r $results_path/intermediate_results/stitched_roof_mask.png -o $results_path/intermediate_results/stitched_object_mask.png -t $results_path/final_results/final_results_roof_layout_estimation.txt
echo "Done!"
cd ..