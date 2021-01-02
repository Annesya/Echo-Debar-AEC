%% Data Preparation
MaxItr = 100;
trainNum = zeros(MaxItr,1);
prev = 0;
for loopNum =1:MaxItr
fprintf(['Count Number: ',num2str(loopNum),'\n']);
num = randi([0 9999],1);
trainNum(loopNum,1) = num;
load("DataAnnotationName");
x_name = table2array(DataAnnotation(num,1));
y_name = table2array(DataAnnotation(num,2));
s_name = table2array(DataAnnotation(num,4));
d_name = table2array(DataAnnotation(num,3));
% Read Data
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\synthetic\farend_speech\')
[x,f] = audioread(x_name);
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\synthetic\echo_signal\')
[y,f] = audioread(y_name);
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\synthetic\nearend_speech\')
[s,f] = audioread(s_name);
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\synthetic\\nearend_mic_signal\')
[d,f] = audioread(d_name);
% Define the frequency and STFT window overlap
f_new = 16000;
winLen = (16*10^-3)*f_new; % 16ms window
overlap = winLen/2; % 50% overlp
fftLen = winLen*2;
% Obtain the STFT
d_stft = stft(d,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
x_stft = stft(x,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
s_stft = stft(s,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
y_stft = stft(y,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);

phase_all = angle(d_stft); % Preserve the phase
% Obtain the magnitude STFT
d_abs = abs(d_stft);
x_abs = abs(x_stft);
s_abs = abs(s_stft);
y_abs = abs(y_stft);

% Adaptive Fltering
fbin = size(d_abs,1); tidx = size(d_abs,2);
W = ones([fbin,tidx]); %Initialize the noise stft
Alpha = zeros([fbin,tidx]); % Initialize the weight A
b0 = 1.0005; % Parameter Beta Vary from 15 to 30 in steps of 5
err = zeros(fbin,tidx); % Function Gamma
pastFrame = 20;
itr = 1; % Number of iteration
for iteration = 1:itr
for i = 2:tidx
    % Calculate the mean of the previous 10 frames of noise
    Xpast = zeros(fbin,1);
    if (i-pastFrame)<=0
        Xpast = (x_abs(:,i-1).^2);
    else
        for j=1:pastFrame
            Xpast = Xpast+(x_abs(:,i-j).^2);
        end
     end
    err(:,i) = (d_abs(:,i-1)-W(:,i-1).*x_abs(:,i-1));
   % A(:,i) = 1./(1+exp(-b*(g(:,i)-1.5)));
    b = b0*(min(1,var(W(:,i-1).*x_abs(:,i-1))/var(err(:,i))));
    Alpha(:,i) = err(:,i)./Xpast;
    W(:,i) = W(:,i-1)+b.*(Alpha(:,i)).*x_abs(:,i);
end
end
upBound = 15;
lowBound = -15;
W1 = zeros(size(W,1),size(W,2));
for i=1:size(W,1)
    for j=1:size(W,2)
        if W(i,j)>upBound
            W1(i,j) = upBound;
        elseif W(i,j)<lowBound
             W1(i,j) = lowBound;
        else
             W1(i,j) = W(i,j);
        end
    end
end
y_estimated_af = W1.*x_abs;
% s_wiener = wiener2(d_abs,[3 3],y_estimated_af);
s_wiener = d_abs - y_estimated_af;
input1 = s_wiener;
numFeatures = size(input1,1); tind = size(input1,2); 
strt = prev+1;
endd = strt+tind-1;
modelIn(:,strt:endd) = input1;
output = s_abs; 
modelTarget(:,strt:endd) = output;
prev = endd;
end
%% Training
featureF = size(modelIn,1);
trainX = reshape(modelIn,size(modelIn,1),1,size(modelIn,2));
trainX = squeeze(num2cell(trainX,[1 2]));
trainY = reshape(modelTarget,size(modelTarget,1),1,size(modelTarget,2));
trainY = squeeze(num2cell(trainY,[1 2]));
layers = layerNew(featureF);
%%
miniBatchSize = 64;
options = trainingOptions("adam", ...
"MaxEpochs",15, ...
"InitialLearnRate",1e-3,...
"MiniBatchSize",miniBatchSize, ...
"Shuffle","every-epoch", ...
"Verbose",true, ...
"Plots","training-progress", ...
"LearnRateSchedule","piecewise", ...
"LearnRateDropFactor",0.9, ...
"LearnRateDropPeriod",1,...
"ExecutionEnvironment","gpu");

aecNetFullyConnected = trainNetwork(trainX,trainY,layers,options);
save('AEC_AF_Wiener_RNN_2','aecNetFullyConnected');
%% Function Definition
function lgraph = layerNew(featureF)
lgraph = layerGraph();
tempLayers = sequenceInputLayer(featureF,"Name","sequence");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = lstmLayer(featureF,"Name","lstm_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = lstmLayer(featureF,"Name","lstm_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    lstmLayer(1024,"Name","lstm_3")
    fullyConnectedLayer(2048,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(featureF,"Name","fc_2")
    reluLayer("Name","relu_2")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"sequence","lstm_1");
lgraph = connectLayers(lgraph,"sequence","addition_1/in2");
lgraph = connectLayers(lgraph,"lstm_1","addition_1/in1");
lgraph = connectLayers(lgraph,"addition_1","lstm_2");
lgraph = connectLayers(lgraph,"addition_1","addition_2/in2");
lgraph = connectLayers(lgraph,"lstm_2","addition_2/in1");

clear tempLayers;
end