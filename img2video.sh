#!/bin/bash
 
python demo/npy.py 

if [ $# -eq 0 ]; then
    python OF_render.py --modality touch

else
	object=$1
	
	mkdir results/output_${object}
	rm -r results/output_${object}/*
	
	python OF_render.py --modality touch --object_file_path ObjectFolder1-100/${object}/ObjectFile.pth --touch_results_dir results/output_${object}
fi

echo "tactile image generated" 

ffmpeg -framerate 5 -i results/output_${object}/%00d.png -c:v libx264 -r 30 -pix_fmt yuv420p results/output_${object}/tactile.mp4

echo "tactile video generated" 
