#!/bin/bash

MODES=$1

case $MODES in
  caltech)
	echo "Demo caltech"
	./tools/demo_detection_caltech.py --gpu 0 \
	;;
  kaist-rgb)
	echo "Demo kaist"
	./tools/demo_detection_kaist_color.py --gpu 0 \
	;;
  kaist-fusion)
	echo "Demo kaist fusion"
	./tools/demo_detection_kaist_fusion.py --gpu 0 \
	;;
  *)
	echo "No mode given"
	exit
	;;
esac
