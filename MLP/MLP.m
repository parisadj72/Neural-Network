clear;
close;
clc;
addpath([pwd '/utils/']); 

% initialize network design and set parameters
model =  struct;
model.numblocks = 5000; 
model.numinitials = 10;
model.weightrange = 1;
model.numhiddenunits = 1;
model.learningrate = 0.1;
model.outputrule = 'sigmoid';

	
% 	inputs and targets will be XOR
model.inputs = [-1 -1
				 1  1
				-1  1
				 1 -1];

model.targets = [1 0
                 1 0
                 0 1
                 0 1];

weightcenter=0; 
result=struct;
v2struct(model)
rng('shuffle')

numattributes = size(inputs,2);
numtargets = size(targets,2);

training=zeros(numblocks,numinitials);

for modelnumber = 1:numinitials
	
	[inweights,outweights] = getweights(numattributes, numhiddenunits, ...
		numtargets, weightrange, weightcenter);
	
	for blocknumber = 1:numblocks
	   
		[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
			FORWARDPASS(inweights,outweights,inputs,outputrule);
		
        % MSE
		accuracy = sum((outputactivations - targets).^2, 2);
		training(blocknumber,modelnumber) = mean(accuracy);		
		
		%   Back-propagating the activations
		[outweights, inweights] = BACKPROP(outweights,inweights,...
			outputactivations,targets,hiddenactivation,...  
			hiddenactivation_raw,inputswithbias,learningrate);
	end
% 	------ TEST SET CAN GO HERE -------
end

% store performance in the result struct
result.training = mean(training,2);