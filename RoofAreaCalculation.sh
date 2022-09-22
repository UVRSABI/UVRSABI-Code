#!/bin/bash
cd RoofAreaCalculation
results_path=../RoofAreaCalculationResults
LOG=$results_path/log.txt
exec > >(tee $LOG)

touch $results_path/out.log
echo "Estimating the roof masks"
cd ../utils/LEDNet/test
chmod 777 test.py
rm -rf RoofMasks
python test.py --datadir $3 --resultdir ../../../RoofAreaCalculation/RoofMasks 
cd ../../../RoofAreaCalculation
echo "Done!"

echo "Displaying Roof Mask Results"
chmod 777 displaymasks.py
python displaymasks.py -i $3 -r RoofMasks -s $results_path/intermediate_results >> $results_path/out.log
echo "Done!"

echo "Estimating the roof area"
python find_area.py --roofmasks RoofMasks --log_file $2 --save_dir_intermediate $results_path/intermediate_results --save_dir_final $results_path/final_results >> $results_path/out.log