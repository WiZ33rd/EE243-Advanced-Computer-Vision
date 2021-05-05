function bbox = getDetections(D)

% D is a sum of difference image
% bbox is a N x 4 matrix, containing the x,y,w,h of each bbox and N is the number of bbox

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.
    ima1 = D;  
    %double the image
    level=graythresh(ima1);
    ima2_b=imbinarize(ima1,level);  
    %remove the noise
    rN = strel('disk',2);
    ima3_rN=imopen(ima2_b,rN);  
    %get the basic information of the Area, Centroid, and BoundingBox   
    img_reg = regionprops(ima3_rN,  'area', 'boundingbox');  
    areas = [img_reg.Area];  
    bbox = cat(1,  img_reg.BoundingBox);    %postion of the bbox 
