function [feat, boxhrange, boxwrange]= getFeatures(I, bbox)

% I is an image
% bbox is a N x 4 matrix, containing the x,y,w,h of each bbox and N is the number of bbox
% feat is a N x n_feat dimensional matrix where n_feat is the feature length
% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS. 

    bins = 16;
    I = double(I);
    feat = zeros(size(bbox,1),bins-1);
    bbox = round(bbox); 
    for m = 1:size(bbox,1)
        boxhrange = bbox(m,2):min(bbox(m,2)+bbox(m,4),size(I,1)); % boundingbox height range
        boxwrange = bbox(m,1):min(bbox(m,1)+bbox(m,3),size(I,2)); % boundingbox width range
        N = hog(I(boxhrange,boxwrange),bins);
        feat(m,:)=N';   % get the features
    end
    % implement the HoG transform for boundingbox
        function N = hog(I,bins)
            [Ix,Iy] = imgradientxy(I);
            phase = angle(Ix+Iy*1i)*180/pi;
            edges = -180:360/bins:180;
            N = histcounts(phase(:),edges);
            N(end-1)=N(end-1)+N(end);
            N(end)=[];
            N = N-mean(N); % normalization 
            N = N/norm(N);
        end
    end