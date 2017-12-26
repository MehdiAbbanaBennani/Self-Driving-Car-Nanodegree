#!/bin/bash
# for f in ./* 
# do
# crop_image $f 1 1 54 0
# done

for f in ./IMG/* 
do
convert $f -resize 200x80! -quality 100 $f
done
