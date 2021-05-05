function new_theta = apply_gradients(x,l,theta,lr)

% x is a matrix of size n_samples x n_feature
% l is a vector of size n_samples x n_class
% theta is a matrix of size n_feature x n_class
% lr is the learning rate

% FILL IN
    % based on reading material linear models of regresssion
    sigmoid=1./(1+exp(-(x*theta)));
    grad=x'*(l-sigmoid);
    new_theta = theta + lr * grad; 
end