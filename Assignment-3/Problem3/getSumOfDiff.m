function sod = getSumOfDiff(I)

% I is a 3D tensor of image sequence where the 3rd dimention represents the time axis

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.
    [x,y,N]=size(I);
    D = zeros(x,y,'uint8');
    for i = 1:N-1
        for j = i+1:N
            D = D + abs(I(:,:,i)-I(:,:,j));
        end
    end
    sod=(2*D)/(N*N-1);