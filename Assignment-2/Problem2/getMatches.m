function matches = getMatches(featI,featR)

% featI and featR are two feature matrices of dim N1 x n_feat and N2 x n_feat respectively.
% matches is a N x 2 matrix indicating the indices of matches. N <= % min(N1,N2). 
% For e.g. if featI(i,:) matches with featR(j,:), then matches should have [i,j] as one of its row.

% FILL IN YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.

% clear all; close all; clc
% 
% 
% I = imread('blocks.png');  % Original image
% R = imread('blocks_tform1.png'); %imwarp(I,tform1);      % Transformed image
% % Corner extraction
% numpoints = 75;
% cornersI = getCorners(I,numpoints); 
% cornersR = getCorners(R,numpoints); 
% 
% % HoG feature extraction
% featI = getFeatures(I,cornersI);
% featR = getFeatures(R,cornersR);


[a,b] = size(featI);
[c,d] = size(featR);

k=1;
matches = zeros(1,2);
for i = 1:a
     x= featI(i,:);
    for j = 1:c
        y = featR(j,:);
        corr(k) = sum( (x-mean(x)).*(y-mean(y)) / (sqrt( var(x)*var(y) )) );
        if corr > 13.5
            matches(k,1)=i;
            matches(k,2)=j;
            k=k+1;
        end
    end
end
end