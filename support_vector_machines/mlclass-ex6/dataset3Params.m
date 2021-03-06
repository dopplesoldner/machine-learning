function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

samples = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

m = size(samples, 1);
errors = zeros(m, m);

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% train for all values of C and Sigma
for i = 1:m
	C = samples(i);
	for j = 1:m
		sigma = samples(j);
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));		%train model
		
		predictions = svmPredict(model, Xval);  %get predicitons using the trained model
		errors(i, j) = mean(double(predictions ~= yval));
	end
end

[row,col] = find(errors == min(errors(:)), 1);

C = samples(row)
sigma = samples(col)

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% =========================================================================

end
