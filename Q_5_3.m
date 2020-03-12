clc
clear
%% Alef

%% Load TrainSet
direction = fileparts(which('Q_5_2.m'));
file_name = [dir('TrainSet\0\*.wav'),dir('TrainSet\1\*.wav'),...
    dir('TrainSet\2\*.wav'),dir('TrainSet\3\*.wav'),...
    dir('TrainSet\4\*.wav'),dir('TrainSet\5\*.wav'),...
    dir('TrainSet\6\*.wav'),dir('TrainSet\7\*.wav'),...
    dir('TrainSet\8\*.wav'),dir('TrainSet\9\*.wav')];

trainSet = cell(size(file_name));
a = cell([size(trainSet,1)*size(trainSet,2) 50]);

classes_lable = zeros(size(trainSet,1)*size(trainSet,2),10);
train_target = zeros(1,size(trainSet,1)*size(trainSet,2));

for i = 1:size(trainSet,2)
    for j = 1:size(trainSet,1)
        trainSet{j,i} = audioread([direction,'\TrainSet\',num2str(i-1),'\',file_name(j,i).name]);
        
        sz = size(trainSet{j,i},1);
        ab = repmat(floor(sz/50),[1 50]);
        ab(end) = floor(sz/50)+(sz-50*floor(sz/50));
        b = trainSet{j,i};
        a((i-1)*(size(trainSet,1))+j,:) = mat2cell(b,ab);
        classes_lable((i-1)*(size(trainSet,1))+j,i) = 1;
        train_target(1,(i-1)*(size(trainSet,1))+j) = i;
    end
end

%% Load TestSet

file_name_tst = [dir('TestSet\0\*.wav'),dir('TestSet\1\*.wav'),...
dir('TestSet\2\*.wav'),dir('TestSet\3\*.wav'),...
dir('TestSet\4\*.wav'),dir('TestSet\5\*.wav'),...
dir('TestSet\6\*.wav'),dir('TestSet\7\*.wav'),...
dir('TestSet\8\*.wav'),dir('TestSet\9\*.wav')];

testSet = cell(size(file_name_tst));
a_tst = cell([size(testSet,1)*size(testSet,2) 50]);

test_target = zeros(1,size(testSet,1)*size(testSet,2));

for i = 1:size(testSet,2)
    for j = 1:size(testSet,1)
        testSet{j,i} = audioread([direction,'\TestSet\',num2str(i-1),'\',file_name_tst(j,i).name]);
        
        sz_tst = size(testSet{j,i},1);
        ab_tst = repmat(floor(sz_tst/50),[1 50]);
        ab_tst(end) = floor(sz_tst/50)+(sz_tst-50*floor(sz_tst/50));
        b_tst = testSet{j,i};
        a_tst((i-1)*(size(testSet,1))+j,:) = mat2cell(b_tst,ab_tst);
        test_target(1,(i-1)*(size(testSet,1))+j) = i;
    end
end

%% Power
power = zeros(size(trainSet,1)*size(trainSet,2),50);
power_tst = zeros(size(testSet,1)*size(testSet,2),50);
for i = 1:size(power,1)
    for j = 1:size(power,2)
        power(i,j) = mean(a{i,j}.*a{i,j});
    end
end
for i = 1:size(power_tst,1)
    for j = 1:size(power_tst,2)
        power_tst(i,j) = mean(a_tst{i,j}.*a_tst{i,j});
    end
end

%% Zero Crossing Rate
ZCR = zeros(size(trainSet,1)*size(trainSet,2),50);
ZCR_tst = zeros(size(testSet,1)*size(testSet,2),50);
for i = 1:size(ZCR,1)
    for j = 1:size(ZCR,2)
        A = ((a{i,j}>=0)-0.5)*2;
        ZCR(i,j) = sum(abs(A-circshift(A,1)));
    end
end
for i = 1:size(ZCR_tst,1)
    for j = 1:size(ZCR_tst,2)
        A = ((a_tst{i,j}>=0)-0.5)*2;
        ZCR_tst(i,j) = sum(abs(A-circshift(A,1)));
    end
end

%% ANNetwork
P = [power,ZCR]';
T = classes_lable';
S = 5;
P_tst = [power_tst,ZCR_tst]';

net = feedforwardnet(S,'trainrp');% OR  net = newff(P,T,S);

net.divideParam.trainRatio = 1.0; % training set
net.divideParam.valRatio = 0.0;
net.divideParam.testRatio = 0.0;

net.trainParam.epochs = 5000;
net.trainParam.lr = 0.1;

net = train(net,P,T);
train_result = net(P);% OR  sim(P)
[~ , train_result] = max(train_result);

accuracy_train = mean((train_result-train_target)==0);

test_result = net(P_tst);% OR  sim(P_tst)
[~ , test_result] = max(test_result);

accuracy_test = mean((test_result-test_target)==0);


