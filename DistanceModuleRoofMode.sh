#!/bin/bash
#!/usr/bin/env python
# conda init bash
cd ./DistanceModuleRoofMode
results_path=../DistanceModuleRoofModeResults
LOG=$results_path/log.txt
exec >> >(tee $LOG)
sudo rm -rf images
mkdir -m 777 images/
echo "Extracting images from video"
python3 ../utils/VideoToImage.py --video $1
echo "Done!!"
cd ..
echo "COLMAP Started"
DATASET_PATH=./DistanceModuleRoofMode #can change it depending on where to keep the bash script.
colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images
colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db
mkdir -m 777 $DATASET_PATH/sparse
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
mkdir -m 777 $DATASET_PATH/dense
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000
colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true
colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply
colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply
colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply
echo "Done!!"
cd ./DistanceModuleRoofMode
echo "Calculations Started"
chmod +x TopViewFinal.py
python3 TopViewFinal.py --logpath $2
echo "Done!!"
