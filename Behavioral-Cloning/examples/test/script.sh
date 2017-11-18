#!/bin/bash
cd IMG/
for f in ./* 
do
crop_image $f 1 1 54 0
done

for f in ./* 
do
if [[ $f == *"cropped"* ]]; then
  convert $f -resize 200x66 $f
fi
done
