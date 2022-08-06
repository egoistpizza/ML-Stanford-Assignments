function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); %number of parameters (features)

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X * theta);

J = sum((-y' * log(sigmoid(X*theta))) - ((1 - y)' * log(1 - sigmoid(X*theta)))) / m + ((lambda / (2 * m)) * sum(theta(2).^2 + theta(3).^2));
j = 1;

    for i = 1 : m
        grad(j) = grad(j) + ( h(i) - y(i) ) * X(i,j);
    end

    for j = 2 : n    
        for i = 1 : m
            grad(j) = grad(j) + ( h(i) - y(i) ) * X(i,j); % Change
        end
        grad(j) = grad(j) + lambda * theta(j); % Change
    end


    grad = (1/m) * grad;
% =============================================================

end
