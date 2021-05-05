function corners = getCorners(I, ncorners)

% I is a 2D matrix 
% ncorners is the number of 'top' corners to be returned
% corners is a ncorners x 2 matrix with the 2D localtions of the corners

% FILL IN YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.

    I = im2double(I);
    r = 5;
    N = 5;
    
    [Ix,Iy] = imgradientxy(I,'sobel');
    Ixx = Ix .* Ix;
    Ixy = Ix .* Iy;
    Iyy = Iy .* Iy;
    
    coefficient = fspecial('gaussian', [N,1], 1);
    w = coefficient*coefficient';
    A11 = imfilter(Ixx,w);
    A12 = imfilter(Ixy,w);
    A22 = imfilter(Iyy,w);
    v2 = ((A11+A22)-sqrt( (A11-A22).^2 + 4*A12.^2 ))/2;
    
    
    Lmax = (v2==imdilate(v2, strel('disk',2*r)));
    Lmax(1:N,:) = false;
    Lmax(:,1:N) = false;
    Lmax(end-N:end,:) = false;
    Lmax(:,end-N:end) = false;
    [rows,cols] = find(Lmax);
    vals = v2(Lmax);
    [vals, indices] = sort(vals, 'descend');
    corners = [rows(indices), cols(indices)];
    corners = corners(1:ncorners,:);    
    
end

