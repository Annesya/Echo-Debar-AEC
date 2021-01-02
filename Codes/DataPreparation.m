%% Data Preparation
MaxItr = 50;
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
t_strt_null = 1;
t_end_null = size(s_abs,2);
while s_abs(:,t_strt_null)==0
    t_strt_null = t_strt_null+1;
end
while s_abs(:,t_end_null)==0
    t_end_null = t_end_null-1;
end
d_abs = d_abs(:,t_strt_null:t_end_null);x_abs = x_abs(:,t_strt_null:t_end_null);s_abs = s_abs(:,t_strt_null:t_end_null);


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
%    stftSegments = squeeze(num2cell(stftSegments,[1 2]));
    strt = prev+1;
    endd = strt+size(stftSegments,3)-1;
   modelIn(:,:,strt:endd) = stftSegments;
   output = s_lps; 
   target = reshape(output,size(output,1),1,size(output,2));
%    modelTarget = squeeze(num2cell(target,[1 2]));
    modelTarget(:,:,strt:endd) = target;
    prev = endd;
end
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