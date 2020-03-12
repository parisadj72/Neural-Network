function...
	[outputactivations,hiddenactivation,hiddenactivation_raw,inputswithbias] = ...
		FORWARDPASS(inweights,outweights,...% weight matrices
			inputpatterns,...% activations to be passed through the model
			outputrule) % option for activation rule

numitems=size(inputpatterns,1);

% input and hidden unit propagation
inputswithbias = [ones(numitems,1),inputpatterns]; 
hiddenactivation_raw=inputswithbias*inweights;

% apply hidden node activation rule
hiddenactivation=sigmoid(hiddenactivation_raw);
hiddenactivation=[ones(numitems,1),hiddenactivation];

% get output activation
outputactivations=hiddenactivation*outweights;

if strcmp(outputrule,'sigmoid') % applying sigmoid
	outputactivations=sigmoid(outputactivations);
end