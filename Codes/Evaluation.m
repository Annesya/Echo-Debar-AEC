for loopNum = 1:1
num = randi([0 9999],1);
trainNum(loopNum,1) = num;
load("DataAnnotationName");
x_name = table2array(DataAnnotation(num,1));
y_name = table2array(DataAnnotation(num,2));
s_name = table2array(DataAnnotation(num,4));
d_name = table2array(DataAnnotation(num,3));
%% Read Data
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
phase_all = angle(d_stft); % Preserve the phase
% Obtain the magnitude STFT
d_abs = abs(d_stft);
x_abs = abs(x_stft);
s_abs = abs(s_stft);

% Obtain the log power spectra (LPS)
d_lps = log10(((d_abs).^2));
x_lps = log10(((x_abs).^2));
% Initialize parameters for the classical Echo Estiation
fbin = size(d_abs,1); tidx = size(d_abs,2);
W = ones([fbin,tidx]); %Initialize the noise stft
Alpha = zeros([fbin,tidx]); % Initialize the weight A
b0 = 0.7; % Parameter Beta Vary from 15 to 30 in steps of 5
err = zeros(fbin,tidx); % Function Gamma
pastFrame = 20;
itr = 1; % Number of iteration
%% Noise Estimate: |N(f,k)|^2 = A.|N(f,k-1)|^2 + (1-A).(T/tau)|Y(f,k)|^2 pp.425(pdf) %% Modified update rule
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
    err(:,i) = (d_abs(:,i)-W(:,i-1).*x_abs(:,i-1));
   % A(:,i) = 1./(1+exp(-b*(g(:,i)-1.5)));
    b = b0*(min(1,var(W(:,i-1).*x_abs(:,i-1))/var(err(:,i))));
    Alpha(:,i) = err(:,i)./Xpast;
    W(:,i) = b.*W(:,i-1)+(1-b).*(Alpha(:,i)).*x_abs(:,i);
end
end

%% Maximum Likelihood Based Suppression Rule pp.234(pdf) Xestimated(f,k) = Gain(f,k)Noisy(f,k)
% where Gain(f,k) = 0.5+0.5*sqrt((p-1)/p) where p is the a-posteriori SNR
   y_estimated_af = W.*x_abs;
   s_estimated_af = d_abs - y_estimated_af;
   input = log10(s_estimated_af.^2);
   numFeatures = size(input,1); tind = size(input,2); numSegments = 1;
   input = [input(:,1:numSegments - 1,:), input];
   stftSegments = zeros(numFeatures, numSegments , size(input,2) - numSegments + 1,size(input,3));
   for num1 = 1:size(input,3)
   for index = 1:size(input,2) - numSegments + 1
       stftSegments(:,:,index,num1) = (input(:,index:index + numSegments - 1,num1)); 
   end
   end
   stftSegments = squeeze(num2cell(stftSegments,[1 2]));
   modelIn = stftSegments;
   modelOut = predict(aecNetFullyConnected,modelIn);
   enhancedS = cell2mat(modelOut);
   enhancedS = (reshape(enhancedS,size(s_estimated_af,1),size(s_estimated_af,2))).*d_abs;
   imj = sqrt(-1);
   s_estm = istft(enhancedS.*exp(imj*phase_all),f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
   s_estm = double(real(s_estm));
   
   s_estm_af = istft(s_estimated_af.*exp(imj*phase_all),f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
   s_estm_af = double(real(s_estm_af));

   figure;
   subplot(2,2,1);stft(d,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
   subplot(2,2,2);stft(s,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
   subplot(2,2,3);stft(s_estm,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
   subplot(2,2,4);stft(s_estm_af,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);

end