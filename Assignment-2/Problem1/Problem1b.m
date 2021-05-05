clear all;close all;clc;

% load the image and implement canny edge detector on it
tiffimg = imread('house.tif','tif');
I = tiffimg(:,:,1);
figure;
subplot(1,2,1);imshow(I),title('Original image');
canny_img = edge(I,'Canny',0.4);
subplot(1,2,2);imshow(canny_img),title('Edge detected(Canny) image');

[H,theta,rho] = houghTrans(canny_img);

figure
imshow(imadjust(rescale(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)

peaks = houghP(H);
x = theta(peaks(:,2));
y = rho(peaks(:,1));
plot(x,y,'s','color','black');

%Now is to find the lines
gap = 20; 
minl = 40; 
minl_sq = minl^2;
gap_sq = gap^2;
nlines = 0; 
[y, x] = find(canny_img);
nzeroD = [x, y] - 1;
dd=peaks(:,1);
ff=peaks(:,2);


for k = 1:size(peaks)
    % Get all pixels associated with Hough transform cell.
    x = nzeroD(:,1);
    y = nzeroD(:,2);
    theta_c = theta(ff(k)) * pi / 180;
    rho_xy = x*cos(theta_c) + y*sin(theta_c);
    nrho = length(rho);
    slope = (nrho - 1)/(rho(end) - rho(1));
    rho_b = round(slope*(rho_xy - rho(1)) + 1);
    idx = find(rho_b == dd(k));
    r = y(idx) + 1; 
    c = x(idx) + 1;
    % Compute distance^2 between the point pairs
    xy = [c r]; % x,y pairs in coordinate system with the origin at (1,1)
    diff_xy_sq = diff(xy,1,1).^2;
    dist_sq = sum(diff_xy_sq,2);  
    % Find the gaps larger than the threshold.
    gap_idx = find(dist_sq > gap_sq);
    idx = [0; gap_idx; size(xy,1)];
    for p = 1:length(idx) - 1
        p1 = xy(idx(p) + 1,:); % offset by 1 to convert to 1 based index
        p2 = xy(idx(p + 1),:); % set the end (don't offset by one this time)
        linelength_sq = sum((p2-p1).^2);
        if linelength_sq >= minl_sq  % this is the condition for the lines it should not be too short
            nlines = nlines + 1;
            lines(nlines).point1 = p1;
            lines(nlines).point2 = p2;
            lines(nlines).theta = theta(peaks(k,2));
            lines(nlines).rho = rho(peaks(k,1)); %put ddata in a struct
        end
    end
end
% for here is to plot the lines 
figure;
imshow(canny_img),title('Hough Transform Detect lines'),hold on 
for k=1:length(lines)    
    xy=[lines(k).point1;lines(k).point2];    
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color','green');    
end



function peaks = houghP(H)
    done = false;                       %set the condition to excute the function
    h_1 = H;
    n_h = size(H)/50;                  %get the defult size
    n_h = max(2*ceil(n_h/2) + 1, 1);  % Make sure the nhood size is odd.
    threshold = 0.3 * max(H(:));
    n_hc = (n_h-1)/2;
    peaks = [];
    while ~done
      [dummy max_idx] = max(h_1(:)); %#ok
      [p, q] = ind2sub(size(h_1), max_idx);
      p = p(1); q = q(1);
      if h_1(p, q) >= threshold
          peaks = [peaks; [p q]];  % add the peak to the list
          % Suppress this maximum and its close neighbors.
          p1 = p - n_hc(1); p2 = p + n_hc(1);
          q1 = q - n_hc(2); q2 = q + n_hc(2);
          % Throw away neighbor coordinates that are out of bounds in
          % the rho direction.
          [qq, pp] = meshgrid(q1:q2, max(p1,1):min(p2,size(H,1)));
          pp = pp(:); qq = qq(:);
          % For coordinates that are out of bounds in the theta
          % direction, we want to consider that H is antisymmetric
          % along the rho axis for theta = +/- 90 degrees.
          theta_too_low = find(qq < 1);
          qq(theta_too_low) = size(H, 2) + qq(theta_too_low);
          pp(theta_too_low) = size(H, 1) - pp(theta_too_low) + 1;
          theta_too_high = find(qq > size(H, 2));
          qq(theta_too_high) = qq(theta_too_high) - size(H, 2);
          pp(theta_too_high) = size(H, 1) - pp(theta_too_high) + 1;  
          % Convert to linear indices to zero out all the values.
          h_1(sub2ind(size(h_1), pp, qq)) = 0;
          done = size(peaks,1) == 20;
      else
          done = true;
      end
    end
end

% Hough transform
function [ H, T, R ] = houghTrans(I)

    [M, N] = size(I);
    thetamax = 90;
    rhomax = floor(sqrt(M^2 + N^2)) - 1; %diagnol length of the image
    T = -thetamax:thetamax - 1; %limitation on theta [-90,89]
    R = -rhomax:rhomax;
    H = zeros(length(R), length(T));

    for m = 1:M
        for n = 1:N
            if I(m, n) > 0 %only find: pixel > 0
                x = n - 1;
                y = m - 1;
                for theta = T
                    rho = round((x * cosd(theta)) + (y * sind(theta)));  %approximate
                    rho_index = rho  + rhomax + 1;
                    theta_index = theta + thetamax + 1;
                    H(rho_index, theta_index) = H(rho_index, theta_index) +1;
                end
            end
        end
    end
end
