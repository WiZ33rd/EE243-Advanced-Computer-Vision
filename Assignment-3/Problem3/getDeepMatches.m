function matches = getDeepMatches(featI,featR)

% featI and featR are two feature matrices of dim N1 x n_feat and N2 x n_feat respectively.
% matches is a N x 2 matrix indicating the indices of matches. N <= min(N1,N2)

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.


matches = [];

len = size(featI,1);
thresholdvalue = 0.7;  

for i = 1: len
    [m,n] = max(featI(i,:)*featR(:,:)');
    if m > thresholdvalue
        matches = [matches;[i,n]]; % match the maximum features
    end
end

end
