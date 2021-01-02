%%  Prepare the training and validation data
load("DataAnnotationName");
% Specify the number of training data for each label 
trainingDataNum = 1000;
% Specify the number of validation data for each label 
validationDataNum = 200;
%Randomised data selection
arr_train = (randperm(10000,trainingDataNum))';
arr_valid = (randperm(10000,validationDataNum))';
% Total number (Train+Validation) of data in each lable
total_data = trainingDataNum + validationDataNum;
% Define the Label Categorical vectors
singleLabels=repelem(categorical("Single_Talk"),total_data,1);
doubleLabels=repelem(categorical("Double_Talk"),total_data,1);

% Define length of Audio segment to be used
f1 = 16000; %sampling rate
len_s = 10; %length in second
data_len = f1*len_s; %sampling rate * time(s)
% Create the data matrix

% Single Talk (Echo only) Dataset
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\synthetic\echo_signal\')
%Training Data
train_single = zeros(data_len,trainingDataNum);

for i=1:trainingDataNum
    j = arr_train(i); 
    [y_in,f]=audioread(table2array(DataAnnotation(j,2)));
    y_in = resample(y_in,f1,f);
    y = zeros(data_len,1);
    if size(y_in,1)< data_len
        y(:,1) = [y_in(:,1)
     zeros((data_len-size(y_in,1)),1)];
    end
    train_single(:,i)= y(1:data_len,1);
end
     %Validation Data
valid_single= zeros(data_len,validationDataNum);
for i=1:validationDataNum
    j = arr_valid(i); 
    [y_in,f]=audioread(table2array(DataAnnotation(j,2)));
    y_in = resample(y_in,f1,f);
    y = zeros(data_len,1);
    if size(y_in,1)< data_len
        y(:,1) = [y_in(:,1)
     zeros((data_len-size(y_in,1)),1)];
    end
    valid_single(:,i)= y(1:data_len,1);
end

% Double Talk (Mic Signal) Dataset
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\synthetic\\nearend_mic_signal\')
%Training Data
train_double= zeros(data_len,trainingDataNum);

for i=1:trainingDataNum
    j = arr_train(i); 
    [y_in,f]=audioread(table2array(DataAnnotation(j,3)));
    y_in = resample(y_in,f1,f);
    y = zeros(data_len,1);
    if size(y_in,1)< data_len
        y(:,1) = [y_in(:,1)
     zeros((data_len-size(y_in,1)),1)];
    end
    train_double(:,i)= y(1:data_len,1);
end
     %Validation Data
valid_double= zeros(data_len,validationDataNum);
for i=1:validationDataNum
    j = arr_valid(i); 
    [y_in,f]=audioread(table2array(DataAnnotation(j,3)));
    y_in = resample(y_in,f1,f);
    y = zeros(data_len,1);
    if size(y_in,1)< data_len
        y(:,1) = [y_in(:,1)
     zeros((data_len-size(y_in,1)),1)];
    end
    valid_double(:,i)= y(1:data_len,1);
end
%%
% Training Preparation
fs = f1 ; % Sampling frequency

% Training Data
audioTrain = [train_single,train_double];
labelsTrain = [singleLabels(1:trainingDataNum);doubleLabels(1:trainingDataNum)];

% Validation Data
audioValidation = [valid_single,valid_double];
labelsValidation = [singleLabels(trainingDataNum+1:end);doubleLabels(trainingDataNum+1:end)];
%% CONVENTIONAL FEATURE
% Define Feature Vector

 aFE = audioFeatureExtractor(...
            "SampleRate",fs, ...
            "Window",hamming(round(0.1*fs),"periodic"), ...
            "OverlapLength",round(0.05*fs), ...
            "melSpectrum",true,...
            "mfcc",true,...
            "mfccDelta",true,...
            "mfccDeltaDelta",true,...
            "gtcc",true,...
            "gtccDelta",true,...
            "gtccDeltaDelta",true);

% Extract Feature from Training Data
featuresTrain = extract(aFE,audioTrain);
featuresValidation = extract(aFE,audioValidation);

%% Input Features Creation
inputFeatures_Train = featuresTrain;
[~,numFeatures,~] = size(inputFeatures_Train);
inputFeatures_Validation = featuresValidation;

inputFeatures_Train = permute(inputFeatures_Train,[2,1,3]);
inputFeatures_Train = squeeze(num2cell(inputFeatures_Train,[1,2]));    
inputFeatures_Validation = permute(inputFeatures_Validation,[2,1,3]);
inputFeatures_Validation = squeeze(num2cell(inputFeatures_Validation,[1,2]));

%%
% Define the DNN Layers
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(512,"OutputMode","sequence")
    bilstmLayer(512,"OutputMode","sequence")
    bilstmLayer(512,"OutputMode","last")
    fullyConnectedLayer(numel(unique(labelsTrain)))
    softmaxLayer
    classificationLayer];

options = trainingOptions("adam", ...
    "InitialLearnRate",1e-4,...
    "MaxEpochs",8,...
    "MiniBatchSize",4,...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropFactor",0.1,...
    "LearnRateDropPeriod",5,...
    "SequenceLength","Shortest",...
    "ExecutionEnvironment","gpu",...
    "Verbose",false);

% Train the Network
talkDetectNet = trainNetwork(inputFeatures_Train,labelsTrain,layers,options);
reset(gpuDevice(1));
%%
for n = 1:validationDataNum
dnn_resp(:,n)=predict(talkDetectNet,inputFeatures_Validation(n));
end
classes = ["Single_Talk","Double_Talk-19"];

[~,classIdx] = max(dnn_resp,[],2);
dnn_PredictedLabels = classes(classIdx);
dnn_PredictedLabels = categorical(dnn_PredictedLabels);

figure
cm = confusionchart(labelsValidation,dnn_PredictedLabels,'title','Test Accuracy - Deep Bi-Directional LSTM Network');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
figure
plotconfusion(labelsValidation,dnn_PredictedLabels');
    