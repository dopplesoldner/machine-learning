function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

y_vec = [1:num_labels] == y;

prod = -y_vec .* log(a3) - ((1 - y_vec) .* log(1 - a3));

% Compute the cost function without regularization
J = sum(sum(prod, 2)) / m;

% Regulisation terms - ignoring the bias terms (column 1)
t1 = [zeros(size(Theta1, 1), 1) Theta1(:, 2:input_layer_size + 1)]; 
t2 = [zeros(size(Theta2, 1), 1) Theta2(:, 2:hidden_layer_size + 1)]; 

reg_terms = (sum(sumsq(t1)) + sum(sumsq(t2))) * (lambda / (2 * m));

J = J + reg_terms;

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for i = 1:m
	yi = y_vec(i, :)'; % 10 x 1 vector
	a1i = a1(i, :)'; % 401 x 1 vector
	
	a2i = a2(i, :)'; % 26 x 1 vector
	z2i = z2(i, :)'; % 25 x 1 vector
	a3i = a3(i, :)'; % 25 x 1 vector
	z3i = z3(i, :)'; % 10 x 1 vector
	
	d3 = a3i - yi; % 10 x 1 vector
	d2 = (Theta2(:, 2:end)' * d3) .* sigmoidGradient(z2i); %25 x 1 vector
	
	D2 = D2 + d3 * a2i';
	D1 = D1 + d2 * a1i';
end;

Theta1_grad = (D1 / m) + (lambda / m) * t1;
Theta2_grad = (D2 / m) + (lambda / m) * t2;


% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
