#!/bin/sh
source ~/.conda-activate
conda activate pyg
export WORKDIR=`pwd`
export PYTHONPATH=$PYTHONPATH:$HOME/libPython
echo WORKDIR=${WORKDIR}
