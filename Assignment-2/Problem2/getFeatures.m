function feat = getFeatures(I, corners)

% I is an image
% corners is a N x 2 matrix with N 2D corner point coordinates
% feat is a N x n_feat matrix where n_feat is the feature dimension of a point

% FILL IN YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS. 

I = im2double(I);
step = 8;

fy = [-1,0,1];
fx = fy';
%     Gx = imfilter(I, fx);
%     Gy = imfilter(I, fy);
%     G=sqrt(Gx.^2+Gy.^2);
    
%     angle = atan(Gy./Gx);
    
    bin = [-pi/2:pi/16:pi/2];
    feat = [];
    for i =1:length(corners)
        x = corners(i,1);
        y = corners(i,2);
        win = I(x-3:x+4,y-3:y+4);
        Gx = imfilter(win, fx);
        Gy = imfilter(win, fy);
        G=sqrt(Gx.^2+Gy.^2);
        angle = atan(Gy./Gx);
        hist=zeros(1,16);
        for k = 1:step
            for l = 1:step
                for u = 1:16
                    if( angle(k,l)>=bin(u) && angle(k,l)<bin(u+1) )
                        hist(u) = hist(u)+G(k,l);
                    end
                end          
            end
        end
        feat = [feat; hist];
    end
        
end