%DFT and DCT transform of 'gonzalezwoods725.png'
img = imread('gonzalezwoods725.png');
grayimg=rgb2gray(img);

%DFT transform of 'gonzalezwoods725.png'
figure;
imshow(img);title('Original image');
DFT_img=fft2(grayimg);
figure;
imshow(DFT_img);title('DFT transformed image');

%DCT transform of 'gonzalezwoods725.png'
figure;
DCT_img=idct2(grayimg);
imshow(DCT_img);title('DCT transformed image');