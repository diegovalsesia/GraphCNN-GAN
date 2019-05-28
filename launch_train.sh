#!/bin/bash

models=("gconv_up_aggr")
pc_classes=("chair")

this_folder=`pwd`

for class_name in ${pc_classes[@]}; do	
	for model in ${models[@]}; do
		
		render_dir="$this_folder/Results/$model/$class_name/renders/"
		log_dir="$this_folder/log_dir/$model/$class_name/"
		save_dir="$this_folder/Results/$model/$class_name/saved_models/"
		CUDA_VISIBLE_DEVICES=0 python "$model""_code/main.py" --class_name $class_name --start_iter $start_iter --render_dir $render_dir --log_dir $log_dir --save_dir $save_dir

	done
done
