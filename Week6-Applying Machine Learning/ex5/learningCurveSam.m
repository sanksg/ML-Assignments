function [error_train, error_val] = ...
    learningCurveSam(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
size(X)

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

numRuns = 10;
	
for samSize = 1:m
	
	JtrainTot = 0;
	JvalTot = 0;

	for rpts = 1:numRuns
		
		randX = randint(samSize,1,length(X))
		randXVal = randint(samSize,1,length(Xval))
		
		Xi = X(randX)
		yi = y(randX)
		
		Xvali = Xval(randXVal)
		yvali = yval(randXVal)
		
		theta = trainLinearReg(Xi, yi, lambda)
		
		htrain = Xi*theta;
		JtrainTot = JtrainTot + (1/(2*length(Xi))) .* (htrain - yi)' * (htrain - yi);
	
	
		hval = Xvali * theta;
		JvalTot = JvalTot + (1/(2*length(Xvali))) .* (hval - yvali)' * (hval - yvali);
	end
	
	error_train(samSize) = JtrainTot/numRuns;
	error_val(samSize) = JvalTot/numRuns;
end
