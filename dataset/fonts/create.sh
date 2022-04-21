#!/bin/bash

set -e

mkdir -p fonts-normal
mkdir -p fonts-all

## copy freefont fonts
cp freefont-20100919/*ttf fonts-all

## copy mac fonts
cp /System/Library/Fonts/Supplemental/Arial* fonts-all
cp /System/Library/Fonts/Supplemental/Times\ New\ Roman* fonts-all
cp /System/Library/Fonts/Supplemental/Verdana* fonts-all

## remove italic from fonts-normal
cp fonts-all/* fonts-normal
rm fonts-normal/*Italic*
rm fonts-normal/*Oblique*
