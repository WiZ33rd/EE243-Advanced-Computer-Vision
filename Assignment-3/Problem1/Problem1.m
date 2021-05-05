clc; close all; clear all;

% read the image
I1 = imread('house.tif');    
I2 = imread('peppers_color.tif');

I1 = I1(:,:,1);
I2 = I2(:,:,1);

% figure,
% subplot(121),imshow(I1);
% subplot(122),imshow(I2);

% r=2;
% sigmaI = 2;
% sigmaX = 4;

SegI1 = normalizedcut(I1,1.91,2,2); % try different sigmaX and sigmaI to see the result 
SegI2 = normalizedcut(I2,1.9,1.8,1);

figure,
subplot(221),imshow(I1);title('Original')
subplot(222),imshow(SegI1);title('Segmetation');
subplot(223),imshow(I2);title('Original');
subplot(224),imshow(SegI2);title('Segmetation');



function SegI = normalizedcut(I1,sigmaX,sigmaI,r)
[OriginalRow,OriginalCol]=size(I1);
I1 = imresize(I1,[100,100]); % change size to reduce calculation
[Row,Col]=size(I1);
N = Row*Col;
W = sparse(N,N);
D = sparse(N,N);

% construct spatial location matrix
locationM = zeros(Row,Col,2);
for i=1:Row
    for j=1:Col
        locationM(i,j,1)=i;
        locationM(i,j,2)=j;
    end
end
locationM = reshape(locationM,[N,1,2]);

% construt intensity feature vectors
feature = reshape(I1,[N,1]);



for i = 1:Col
    for j = 1:Row
        % find the range of Column and Rows that satisfy euclidan distance
        rangecol = (i - r) : (i + r);   
        rangerow = ((j - r) :(j + r))';
        sortcol = find(rangecol>0 & rangecol<(Col+1));
        sortrow = find(rangerow>0 & rangerow<(Row+1));
        rangecol = rangecol(sortcol);  
        rangerow = rangerow(sortrow);
        
        % choose each pixel as referrred node to compare with other nodes
        node = j + (i-1)*Row;
        lengthrow = length(rangerow);
        lengthcol = length(rangecol);
        tmp1 = zeros(lengthrow,lengthcol);
        tmp2 = zeros(lengthrow,lengthcol);
        for m = 1:length(rangerow)
            for n = 1:length(rangecol)
                tmp1(m,n) = rangerow(m,1);
                tmp2(m,n) = ((rangecol(1,n)-1).*Row);
            end
        end
        
        % the index of compared nodes in distance range
        comparenode = reshape((tmp1+tmp2),[lengthrow*lengthcol,1]);

        % calculate the similarity of spatial location
        locationSj = zeros(length(comparenode),1,2);
        for n = 1:2
            for m = 1:length(comparenode)
                locationSj(m,1,n) = locationM(comparenode(m,1),1,n);
            end
        end
%         for m = 1:length(comparenode)
%             locationSj(m,1,2) = locationM(comparenode(m,1),1,2);
%         end
        
        locationSi = zeros(length(comparenode),1,2);
        locationtmp = locationM(node,1,:);
        for m =1:length(comparenode)
            for n = 1:2
                locationSi(m,1,n) = locationtmp(1,1,n);
            end
        end
        
        % square euclidan distance
        locationDiff = locationSi-locationSj;
        locationDiff = sum(locationDiff.*locationDiff,3);
        X = (sqrt(locationDiff)<(r+1));
        comparenode = comparenode(X);
        locationDiff = locationDiff(X);
        
        % calculate vector disimilarity
        featureJ = zeros(length(comparenode),1);
        for m = 1:length(comparenode)
            featureJ(m,1) = feature(comparenode(m,1),1);
        end
        featureJ = uint8(featureJ);
        
        featureI = zeros(length(comparenode),1);
        featureItmp = feature(node,1,:);
        for m =1:length(comparenode)
            featureI(m,1) = featureItmp(1,1);
        end
        featureI = uint8(featureI);
        
        % construct W
        featureDiff = featureI-featureJ;
        featureDiff = sum(featureDiff.*featureDiff,3);
        W(node, comparenode)= exp(-featureDiff/sigmaI^2).*exp(-locationDiff/sigmaX^2);
    end
end

% construct matrix D
for i = 1:N
    D(i,i) = sum(W(i,:)); % calculate the diagonal of W
end

% solve the equation (D - W)*Y = L * D * Y
[Y,L] = eigs(D-W, D, 2, 'sm');
smalleig = Y(:,2);
point = median(smalleig); % chose median of second smallest eigenvector
SegI = reshape(smalleig,[Row,Col]);
SegI = imresize(SegI,[OriginalRow,OriginalCol]);
SegI = imbinarize(SegI,point);

end

 
        
        

