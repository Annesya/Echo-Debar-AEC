% Read Data
num = randi([0 9999],1);
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
b0 = 1.0005; % Parameter Beta Vary from 15 to 30 in steps of 5
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
s_estimated_af = d_abs - y_estimated_af;
noisy_abs = s_estimated_af;
%%
% Initialize parameters for the classical Noise Estiation
NPsd = ones([fbin,tidx]); %Initialize the noise stft
X_estm = zeros(fbin,tidx);% Initialize clean speech psd estimate
A = zeros([fbin,tidx]); % Initialize the weight A
b = 20; % Parameter Beta Vary from 15 to 30 in steps of 5
g = zeros(fbin,tidx); % Function Gamma
pastFrame = 10;
tau = 0.5; % update factor
itr = 100; % Number of iteration
%% Noise Estimate: |N(f,k)|^2 = A.|N(f,k-1)|^2 + (1-A).(T/tau)|Y(f,k)|^2 pp.425(pdf) %% Modified update rule
for iteration = 1:itr
for i = 2:tidx
    % Calculate the mean of the previous 10 frames of noise
    Npast = zeros(fbin,1);
    if (i-pastFrame)<=0
        Npast = (NPsd(:,i-1).^2);
    else
        for j=1:pastFrame
            Npast = Npast+(NPsd(:,i-j).^2);
        end
        Npast = Npast/pastFrame;
    end
    g(:,i) = (noisy_abs(:,i).^2)./Npast;
   % A(:,i) = 1./(1+exp(-b*(g(:,i)-1.5)));
    A(:,i) = 1-min(1,(1./(g(:,i).^2)));
    NPsd(:,i) = A(:,i).*(NPsd(:,i-1).^2)+(i/tau)*(1-A(:,i)).*(noisy_abs(:,i).^2);
end
end

%% Maximum Likelihood Based Suppression Rule pp.234(pdf) Xestimated(f,k) = Gain(f,k)Noisy(f,k)
% where Gain(f,k) = 0.5+0.5*sqrt((p-1)/p) where p is the a-posteriori SNR
snr_upBound = 10^3;
apost_snr = (noisy_abs.^2)./NPsd;
apost_snr = min(apost_snr,snr_upBound);
gain = abs((0.5+0.5*sqrt((apost_snr-1)./apost_snr)));
clean_estimated = gain.*noisy_abs;
%% Generate Estimated Y and S
y1 = y_estimated_af;
y1fft = y1.*exp(sqrt(-1)*phase_all(:,1:size(y1,2)));
y1_t = istft(y1fft,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
y_Estm= real(y1_t);

s1 = s_estimated_af;
s1fft = s1.*exp(sqrt(-1)*phase_all(:,1:size(s1,2)));
s1_t = istft(s1fft,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
s_Estm = real(s1_t);

s1 = clean_estimated;
s1fft = s1.*exp(sqrt(-1)*phase_all(:,1:size(s1,2)));
s1_t = istft(s1fft,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
s_Estm_cleaned = real(s1_t);

%% Maximum Likelihood Based Suppression Rule pp.234(pdf) Xestimated(f,k) = Gain(f,k)Noisy(f,k)
% where Gain(f,k) = 0.5+0.5*sqrt((p-1)/p) where p is the a-posteriori SNR
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
%% Plot
figure;
subplot(3,2,1);stft(d,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);
subplot(3,2,2);stft(x,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);
subplot(3,2,3);stft(s,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);
subplot(3,2,4);stft(y,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);
subplot(3,2,5);stft(s_Estm,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);
subplot(3,2,6);stft(y_Estm,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);

