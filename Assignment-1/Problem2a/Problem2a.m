%4×4 DFT and DCT basis image
close all;clear all;clc;
DFT = dftmtx(4);
DCT = dctmtx(4);

figure;
sgtitle('4×4 DFT basis images');
display_img(DFT);

figure;
sgtitle('4×4 DCT basis images');
display_img(DCT);


function display_img(img)
line=4;row=4;p=1;
    for i=1:line
        for j=1:row
            TF_img=img(i,:)'*img(j,:);
            subplot(line,row,p);
            imshow(TF_img);colormap(gray);
         p=p+1;
        end
    end
end