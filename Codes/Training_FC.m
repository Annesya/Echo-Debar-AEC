%% Training
    miniBatchSize = 16;
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
    "ExecutionEnvironment","cpu");
%%
% modelIn1 = reshape(modelIn,size(modelIn,1)/2,size(modelIn,2)*2,size(modelIn,3));
numFeatures = size(modelIn,1); numSegments = size(modelIn,2);
trainX = squeeze(num2cell(modelIn,[1 2]));
trainY = squeeze(num2cell(modelTarget,[1 2]));
trainLayer = layerFormation2(numFeatures,numSegments);
% transLayer = aecNetFullyConnected.Layers(1:end-1);
% trainLayer = [transLayer
%         regressionLayer];
aecNetFullyConnected = trainNetwork(trainX,trainY,trainLayer,options);
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
function lgraph = layerFormation2(numFeatures,numSegments)
bnode = numFeatures*numSegments;
lgraph = layerGraph();
tempLayers = [
    sequenceInputLayer([numFeatures numSegments 1],"Name","sequence")
    flattenLayer("Name","flatten")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = lstmLayer(bnode,"Name","lstm_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = lstmLayer(bnode,"Name","lstm_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    lstmLayer(2048,"Name","lstm_3");
    fullyConnectedLayer(2048,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer((numFeatures/2),"Name","fc_2")
    reluLayer("Name","relu_2")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"flatten","lstm_1");
lgraph = connectLayers(lgraph,"flatten","addition_1/in2");
lgraph = connectLayers(lgraph,"lstm_1","addition_1/in1");
lgraph = connectLayers(lgraph,"addition_1","lstm_2");
lgraph = connectLayers(lgraph,"addition_1","addition_2/in2");
lgraph = connectLayers(lgraph,"lstm_2","addition_2/in1");

clear tempLayers;
end