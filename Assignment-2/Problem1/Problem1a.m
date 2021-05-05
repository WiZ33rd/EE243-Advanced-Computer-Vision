clear all;close all;clc;

% load images
tiffimg = imread('house.tif','tif');
img1 = tiffimg(:,:,1);
img2 = imread('lena_gray_256.tif','tif');
log(img1);canny(img1);
log(img2);canny(img2);

% use Laplace of Gaussian edge detector on each image
% try differrent threshhold for each images
function log(img)
    j=0;
    figure;
    for i=0.001:0.001:0.012
        log_img = edge(img,'log',i);
        j=j+1;
        subplot(3,4,j);imshow(log_img);title(['LOG threshold=',num2str(i)]);
    end
end

% use Canny edge detector on each image
% try differrent threshhold for each images
function canny(img)
    j=0;
    figure;
    for i=0.01:0.01:0.12
        canny_img = edge(img,'Canny',i);
        j=j+1;
        subplot(3,4,j);imshow(canny_img);title(['Canny threshold=',num2str(i)]);
    end
end