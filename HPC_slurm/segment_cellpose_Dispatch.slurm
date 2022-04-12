#!/bin/sh
#  segment_cellpose_Dispatch.slurm
#  
#  Created by Yuriy Alexandrov
#
## sed -i s/\\r//g segment_cellpose_Dispatch.slurm ------- fixes DOS end lines

#SBATCH --nodes=1
#SBATCH --ntasks=16 
#SBATCH --time=1:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-cpu=8gb

# important - always put here a number > (number of your images)
#SBATCH --array=1-32

ARGS="$1"
echo $ARGS

ARRARGS=(${ARGS//:/ })

CODEDIR=${ARRARGS[0]}
CODEFILE=${ARRARGS[1]}
SRCDIR=${ARRARGS[2]}
DSTDIR=${ARRARGS[3]}
IMAGEFILEEXTENSION=${ARRARGS[4]}
MODE=${ARRARGS[5]}
MODELTYPE=${ARRARGS[6]}
CELLPROB_THRESHOLD=${ARRARGS[7]}
FLOW_THRESHOLD=${ARRARGS[8]}
RESAMPLE=${ARRARGS[9]}
DIAMETER=${ARRARGS[10]}
GPU=${ARRARGS[11]}

jno=$SLURM_ARRAY_TASK_ID

echo "Start time $(date)"

umask 22 # make all files group and world readable

module load Anaconda3

conda init bash

source ~/.bashrc

conda activate cellpose

nvcc --version

space_replacing_template=@I@

NIMAGES=0
for f in $SRCDIR/*; do
	case "$f" in
        *$IMAGEFILEEXTENSION*)
	    #echo $(basename "$f")
	    NIMAGES=$((NIMAGES + 1))
	;;
	esac
done
echo $NIMAGES

if (( (($NIMAGES == 0)) )); then
      echo "nothing to do, exiting"
      return
fi

fname="blank"

cnt=0
for f in $SRCDIR/*; do
	case "$f" in
        *$IMAGEFILEEXTENSION*)
	    #echo $(basename "$f")
	    cnt=$(($cnt + 1))
		if [ "x$jno" == "x$cnt" ]; then
			fname=$f
		fi
	;;
	esac
done

echo $fname

if [ "$fname" != "blank" ]; then

  fname="${fname// /$space_replacing_template}"
	
  ARGUMENTS="$DSTDIR:$fname:$MODE:$MODELTYPE:$CELLPROB_THRESHOLD:$FLOW_THRESHOLD:$RESAMPLE:$DIAMETER:$GPU:$space_replacing_template"
    
  python $CODEDIR/$CODEFILE $ARGUMENTS
 	
fi

exit