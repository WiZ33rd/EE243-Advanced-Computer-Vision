clear all;close all;clc;

tiffimg = imread('house.tif', 'tif');
img = tiffimg(:,:,1);
corners=getCorners(img,50);
figure;imshow(img);hold on;
plot(corners(:,2),corners(:,1), '*r');