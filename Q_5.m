clc
clear

%% Load TrainSet
direction = fileparts(which('Q_5.m'));
file_name = [dir('TrainSet\0\*.wav'),dir('TrainSet\1\*.wav'),...
    dir('TrainSet\2\*.wav'),dir('TrainSet\3\*.wav'),...
    dir('TrainSet\4\*.wav'),dir('TrainSet\5\*.wav'),...
    dir('TrainSet\6\*.wav'),dir('TrainSet\7\*.wav'),...
    dir('TrainSet\8\*.wav'),dir('TrainSet\9\*.wav')];

trainSet = cell(size(file_name));
max_set = 0;

classes_lable = zeros(size(trainSet,1)*size(trainSet,2),10);

for i = 1:size(trainSet,2)
    for j = 1:size(trainSet,1)
        [trainSet{j,i} , Fs] = audioread([direction,'\TrainSet\',num2str(i-1),'\',file_name(j,i).name]);
        classes_lable((i-1)*(size(trainSet,1))+j,i) = 1;
        train_target(1,(i-1)*(size(trainSet,1))+j) = i;
        max_set = max(max_set , numel(trainSet{j,i}));
    end
end

%% Load TestSet

file_name_tst = [dir('TestSet\0\*.wav'),dir('TestSet\1\*.wav'),...
dir('TestSet\2\*.wav'),dir('TestSet\3\*.wav'),...
dir('TestSet\4\*.wav'),dir('TestSet\5\*.wav'),...
dir('TestSet\6\*.wav'),dir('TestSet\7\*.wav'),...
dir('TestSet\8\*.wav'),dir('TestSet\9\*.wav')];

testSet = cell(size(file_name_tst));

classes_lable_tst = zeros(size(testSet,1)*size(testSet,2),10);

for i = 1:size(testSet,2)
    for j = 1:size(testSet,1)
        [testSet{j,i} , Fs] = audioread([direction,'\TestSet\',num2str(i-1),'\',file_name_tst(j,i).name]);
        classes_lable_tst((i-1)*(size(testSet,1))+j,i) = 1;
        test_target(1,(i-1)*(size(testSet,1))+j) = i;
        max_set = max(max_set , numel(testSet{j,i}));
    end
end

%% Making Files the Same Length

for i = 1:size(trainSet,2)
    for j = 1:size(trainSet,1)
        trainSet{j,i} = padarray(trainSet{j,i} , [(max_set - numel(trainSet{j,i})) 0] , eps , 'post');
    end
end

for i = 1:size(testSet,2)
    for j = 1:size(testSet,1)
        testSet{j,i} = padarray(testSet{j,i} , [(max_set - numel(testSet{j,i})) 0] , eps , 'post');
    end
end

%% MFCC
z = 1;
for i = 1:size(trainSet,2)
    for j = 1:size(trainSet,1)
        mfcc{j,i} = Speech_MFCC(trainSet{j,i});
        TrainData(z,:) = mfcc{j,i}(:)';
        
        % -------Cepstral Mean Substraction Normalisation:
        mfcc_cms{j,i} = mfcc{j,i} - mean(mfcc{j,i},2);
        TrainData_cms(z,:) = mfcc_cms{j,i}(:)';
        z = z + 1;
    end
end

z = 1;
for i = 1:size(testSet,2)
    for j = 1:size(testSet,1)
        mfcc_test{j,i} = Speech_MFCC(testSet{j,i});
        TestData(z,:) = mfcc_test{j,i}(:)';
        
        % -------Cepstral Mean Substraction Normalisation:
        mfcc_test_cms{j,i} = mfcc_test{j,i} - mean(mfcc_test{j,i},2);
        TestData_cms(z,:) = mfcc_test_cms{j,i}(:)';
        z = z + 1;
    end
end

%% Train and Test the Network

P = TrainData';
T = classes_lable';
S = 3763;
P_tst = TestData';
T_tst = classes_lable_tst';

net = feedforwardnet(S,'traingd');
% net = newff(P,T,S);
net.divideParam.trainRatio = 1.0; % training set
net.divideParam.valRatio = 0.0; % validation set
net.divideParam.testRatio = 0.0; % test set
net.trainParam.epochs = 5000;
net.trainParam.lr = 0.2;

net = train(net,P,T);
train_result = net(P);% OR  sim(P)
[~ , train_result] = max(train_result);
accuracy_train = mean((train_result-T)==0);

% test_result = net(P_tst);% OR  sim(P_tst)
% [~ , test_result] = max(test_result);
% accuracy_test = mean((test_result-T_tst)==0);

%% Results, Confusion Matrix

confusionmatrix_tr = zeros(10,10);
confusionmatrix_te = zeros(10,10);
index = 1;
for i = 1:10
    % Confusion Matrix
    for j = 1:10
        confusionmatrix_tr(i,j) = sum(train_result(index : end) == j)/size(trainSet,1);
    end 
    index = index + size(trainSet,1);
end

% index = 1;
% for i = 1:10
%     % Confusion Matrix
%     for j = 1:10
%         confusionmatrix_te(i,j) = sum(test_result(index : end) == j)/size(testSet,1);
%     end 
%     index = index + size(testSet,1);
% end

%% Train and Test the Network for CMS Normalisation

P = TrainData_cms';
T = classes_lable';
S = 3763;
P_tst = TestData_cms';
T_tst = classes_lable_tst';

net = feedforwardnet(S,'traingd');
% net = newff(P,T,S);
net.divideParam.trainRatio = 1.0; % training set
net.divideParam.valRatio = 0.0; % validation set
net.divideParam.testRatio = 0.0; % test set
net.trainParam.epochs = 5000;
net.trainParam.lr = 0.2;

net = train(net,P,T);
train_result = net(P);% OR  sim(P)
[~ , train_result] = max(train_result);
accuracy_train = mean((train_result-T)==0);
% test_result = net(P_tst);% OR  sim(P_tst)
% [~ , test_result] = max(test_result);
% accuracy_test = mean((test_result-T_tst)==0);

%% Results, Confusion Matrix

confusionmatrix_tr = zeros(10,10);
confusionmatrix_te = zeros(10,10);
index = 1;
for i = 1:10
    % Confusion Matrix
    for j = 1:10
        confusionmatrix_tr(i,j) = sum(train_result(index : end) == j)/size(trainSet,1);
    end 
    index = index + size(trainSet,1);
end
% index = 1;
% for i = 1:10
%     % Confusion Matrix
%     for j = 1:10
%         confusionmatrix_te(i,j) = sum(test_result(index : end) == j)/size(testSet,1);
%     end 
%     index = index + size(testSet,1);
% end

