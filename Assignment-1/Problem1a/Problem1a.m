%2D-DFT

% load the 'house.tif'
img = imread('house.tif','tif');
img = img(:,:,1);

figure;
imshow(img);
title('Original image house');

figure;
title('Magnitude image');
% run fast Fourier transform on the image
img = fftshift(img);
% 2D Discrete Fourier transform on the image
fourier_img = fft2(img);
% move high frequency into center and display in gray scale
imagesc(100*log(1+abs(fftshift(fourier_img)))); colormap(gray); 

figure;
title('Phase spectrum');
imagesc(angle(fourier_img));  colormap(gray);