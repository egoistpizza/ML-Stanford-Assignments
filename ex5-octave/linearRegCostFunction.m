function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

# Regularized Linear Regression Cost
regTerm = lambda/(2*m) * sum(theta(2:end).^2); # Vectorized regularization term

predictions = X * theta;
sqrErrors = (predictions - y).^2 ;
J = (1/(2*m) * sum(sqrErrors)) + regTerm;

# Regularized Linear Regression Gradient
n = length(theta); %number of parameters (features)

h = X * theta

j = 1;

    for i = 1 : m
        grad(j) = grad(j) + ( h(i) - y(i) ) * X(i,j);
    end

    for j = 2 : n    
        for i = 1 : m
            grad(j) = grad(j) + ( h(i) - y(i) ) * X(i,j);
        end
        grad(j) = grad(j) + lambda * theta(j);
    end


    grad = (1/m) * grad;


% =========================================================================

grad = grad(:);

end
