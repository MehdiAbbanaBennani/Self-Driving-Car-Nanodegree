#!/bin/bash
cd images/
for f in ./* 
do
crop_image $f 1 1 54 0
# take action on each file. $f store current file name
done

