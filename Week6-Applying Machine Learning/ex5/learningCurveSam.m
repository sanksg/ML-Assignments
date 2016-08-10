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

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

numRuns = 50;
	
for samSize = 1:m	
	
	JtrainTot = 0;
	JvalTot = 0;

    samSize
	for rpts = 1:numRuns
		rpts
		randX = randi(length(X),samSize, 1);
		randXVal = randi(length(Xval),samSize, 1);
		
		Xtrain = X(randX,:);
		ytrain = y(randX,:);
		
		Xvali = Xval(randXVal,:);
		yvali = yval(randXVal,:);
		
		theta = trainLinearReg(Xtrain, ytrain, lambda);
		
		htrain = Xtrain*theta;
		JtrainTot = JtrainTot + (1/(2*length(Xtrain))) .* (htrain - ytrain)' * (htrain - ytrain);
	
	
		hval = Xvali * theta;
		JvalTot = JvalTot + (1/(2*length(yvali))) .* (hval - yvali)' * (hval - yvali);
	end
	

	error_train(samSize) = JtrainTot/numRuns;
	error_val(samSize) = JvalTot/numRuns;
end
