function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    h = theta' * X';

    for j=1:size(theta),
        derrivative = 0;
        for i=1:m,
            derrivative = derrivative + (h(1, i) - y(i))*(X(i, j));
        end;
        derrivative = derrivative * (1 / m);

        theta(j, 1) = theta(j, 1) - alpha * derrivative;
    end;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
