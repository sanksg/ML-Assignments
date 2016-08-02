function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

display(size(X));
display(size(X'));
s1 = inv(X'*X);
display(size(s1));
s2=s1*X';
display(size(s2));
theta = s2*y;
display(size(theta));



% -------------------------------------------------------------


% ============================================================

end
