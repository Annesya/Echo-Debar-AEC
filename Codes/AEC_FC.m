miniBatchSize = 128;
    options = trainingOptions("adam", ...
    "MaxEpochs",15, ...
    "InitialLearnRate",1e-3,...
    "MiniBatchSize",miniBatchSize, ...
    "Shuffle","every-epoch", ...
    "Verbose",true, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.9, ...
    "LearnRateDropPeriod",1,...
    "ExecutionEnvironment","cpu");
MaxItr = 50;
trainNum = zeros(MaxItr,1);
for loopNum =5:MaxItr
fprintf(['Count Number: ',num2str(loopNum),'\n']);
num = randi([0 9999],1);
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
t_strt_null = 1;
t_end_null = size(s_abs,2);
while s_abs(:,t_strt_null)==0
    t_strt_null = t_strt_null+1;
end
while s_abs(:,t_end_null)==0
    t_end_null = t_end_null-1;
end
d_abs = d_abs(:,t_strt_null:t_end_null);x_abs = x_abs(:,t_strt_null:t_end_null);s_abs = s_abs(:,t_strt_null:t_end_null);

% delta = 10^-4;
% for d1 = 1:size(s_abs,1)
%     for d2 = 1:size(s_abs,2)
%         if s_abs(d1,d2)==0
%             s_abs(d1,d2) = delta;
%             d_abs(d1,d2) = delta;
%             x_abs(d1,d2) = delta;
%         end
%     end
% end
% Obtain the log power spectra (LPS)
d_lps = log10(((d_abs).^2));
x_lps = log10(((x_abs).^2));
s_lps = log10(((s_abs).^2));
input1 = [x_lps
        d_lps];
   numFeatures = size(input1,1); tind = size(input1,2); numSegments = 2;
   input = [input1(:,1:numSegments - 1,:), input1];
   stftSegments = zeros(numFeatures, numSegments , size(input,2) - numSegments + 1,size(input,3));
   for num1 = 1:size(input,3)
   for index = 1:size(input,2) - numSegments + 1
       stftSegments(:,:,index,num1) = (input(:,index:index + numSegments - 1,num1)); 
   end
   end
   stftSegments = squeeze(num2cell(stftSegments,[1 2]));
   modelIn = stftSegments;
   output = s_lps; 
   target = reshape(output,size(output,1),1,size(output,2));
   modelTarget = squeeze(num2cell(target,[1 2]));
   % Model definition
   if loopNum ==1
     layers_new = layerFormation(numFeatures,numSegments);
   else
    transLayer = aecNetFullyConnected.Layers(1:end-3);
    layers_new = transLayerFormation(transLayer,numFeatures);
   end
   aecNetFullyConnected = trainNetwork(modelIn,modelTarget,layers_new,options);
end
save('FC_AEC','aecNetFullyConnected');
%% plot function
t_idx = 1:tind;
f_idx = 1:(numFeatures/2);
figure;
subplot(3,1,1);
waterfall(t_idx,f_idx,x_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;ylim([0 Inf]);
subplot(3,1,2);
waterfall(t_idx,f_idx,d_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;ylim([0 Inf]);
subplot(3,1,3);
waterfall(t_idx,f_idx,s_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;ylim([0 Inf]);
%%
function lgraph = layerFormation(numFeatures,numSegments)
bnode = numFeatures*numSegments/2;
lgraph = layerGraph();
tempLayers = [
    sequenceInputLayer([numFeatures numSegments 1],"Name","sequence")
    flattenLayer("Name","flatten")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = bilstmLayer(bnode,"Name","bilstm_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = bilstmLayer(bnode,"Name","bilstm_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    fullyConnectedLayer(2048,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer((numFeatures/2),"Name","fc_2")
    reluLayer("Name","relu_2")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"flatten","bilstm_1");
lgraph = connectLayers(lgraph,"flatten","addition_1/in2");
lgraph = connectLayers(lgraph,"bilstm_1","addition_1/in1");
lgraph = connectLayers(lgraph,"addition_1","bilstm_2");
lgraph = connectLayers(lgraph,"addition_1","addition_2/in2");
lgraph = connectLayers(lgraph,"bilstm_2","addition_2/in1");

clear tempLayers;
end
function lgraph = transLayerFormation(transLayer,numFeatures)
lgraph = layerGraph();
tempLayers = [
    transLayer
    fullyConnectedLayer((numFeatures/2),"Name","newFC",'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);
lgraph = connectLayers(lgraph,"flatten","addition_1/in2");
% lgraph = connectLayers(lgraph,"addition_1","addition_2/in2");
end