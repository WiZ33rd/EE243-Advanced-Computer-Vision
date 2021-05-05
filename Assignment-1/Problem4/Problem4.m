close all;clear all;clc;
%load the moise image
I = imread('jump_noisy.png');
I = I(:,:,1);

% reduce ==> {2, 4, 8}
I1 = impyramid(I, 'reduce');
I2 = impyramid(I1, 'reduce');
I3 = impyramid(I2, 'reduce');

% show original and decomposed image
figure
a1 = subplot(1, 4, 1);imshow(I),xs = get(a1, 'xlim'); ys = get(a1, 'ylim');title('Original image');
a2 = subplot(1, 4, 2); imshow(I1), set(a2, 'xlim', xs, 'ylim', ys);title('Gaussian 1/2');
a3 = subplot(1, 4, 3); imshow(I2), set(a3, 'xlim', xs, 'ylim', ys);title('Gaussian 1/4');
a4 = subplot(1, 4, 4); imshow(I3), set(a4, 'xlim', xs, 'ylim', ys);title('Gaussian 1/8');

figure;
%Use DFT transform on the noisy image
fourier_img = fftshift(fft2(I3));
DFT_img=log(abs(fourier_img));
imagesc(DFT_img); colormap(jet);colorbar; 
title('DFT image');

noise_ary = DFT_img>10;
figure;
imagesc(noise_ary); colormap(gray);colorbar; title('Binary image of DFT');

% pick noise (middle of DFT map) located in (100:150)
noise_ary(100:150,:) = 0;
figure;
imagesc(noise_ary); colormap(gray);colorbar; title('Noises to remove'); 

% remove noise
fourier_img(noise_ary) = 0;
DFT2_img = log(abs(fourier_img));
min_val = min(min(DFT2_img));
max_val = max(max(DFT2_img));
figure;
imshow(DFT2_img,[min_val,max_val]);title('Noise removed');

% reconstruct image from noise removed DFT image
dis_img=ifft2(fftshift(fourier_img));
filterred_img=abs(dis_img);
min_val = min(min(filterred_img));
max_val = max(max(filterred_img));
figure;
imshow(filterred_img,[min_val,max_val]);

% expand reconstructed image
newI2 = impyramid(filterred_img,'expand');
newI1 = impyramid(newI2,'expand');
newI = impyramid(newI1,'expand');
min_val = min(min(newI));
max_val = max(max(newI));
figure;
imshow(newI,[min_val,max_val]);title('reconstructed image');