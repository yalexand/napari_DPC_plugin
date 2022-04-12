#!/bin/bash
#
#  Created by Yuriy Alexandrov
#

# change as needed
CODEDIR=`pwd` 
CODEFILE=segment_cellpose_image.py

SRCDIR="/camp/lab/frenchp/data/CALM/working/alexany/Collaborators/Huihui_Liu/KV_2022_04_07/DPC_reconstructed/"
DSTDIR="/camp/lab/frenchp/data/CALM/working/alexany/Collaborators/Huihui_Liu/CAMP_integration/output_DPC/"

IMAGEFILEEXTENSION=.tif
MODE=DPC
MODELTYPE=cyto
CELLPROB_THRESHOLD=0.0
FLOW_THRESHOLD=0.0
RESAMPLE=True
DIAMETER=40
GPU=True

ARGS="$CODEDIR:$CODEFILE:$SRCDIR:$DSTDIR:$IMAGEFILEEXTENSION:$MODE:$MODELTYPE:$CELLPROB_THRESHOLD:$FLOW_THRESHOLD:$RESAMPLE:$DIAMETER:$GPU"

echo "launching processing job for $IMAGEFILEEXTENSION"

echo $(sbatch $CODEDIR/segment_cellpose_Dispatch.slurm $ARGS)
