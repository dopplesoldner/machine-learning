function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h_theta_x = sigmoid(X * theta); %hypothesis
J = sum(-y .* log(h_theta_x) - (1 - y) .* log(1 - h_theta_x)) * (1/m);

grad = (X' * (h_theta_x - y)) * (1/ m);

end
