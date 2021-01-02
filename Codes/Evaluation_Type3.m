% Define the frequency and STFT window overlap
x = audioread('_5z9G2AP806bhcI0QF18Qg_doubletalk_lpb.wav');
d = audioread('_5z9G2AP806bhcI0QF18Qg_doubletalk_mic.wav');
f_new = 16000;
winLen = (16*10^-3)*f_new; % 16ms window
overlap = winLen/2; % 50% overlp
fftLen = winLen*2;
% Obtain the STFT
d_stft = stft(d,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
x_stft = stft(x,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);

phase_all = angle(d_stft); % Preserve the phase
% Obtain the magnitude STFT
d_abs = abs(d_stft);
x_abs = abs(x_stft);
% Obtain the log power spectra (LPS)
d_lps = log10(((d_abs).^2));
x_lps = log10(((x_abs).^2));

% Initialize parameters for the classical Echo Estiation
fbin = min(size(d_abs,1),size(x_abs,1)); tidx = min(size(d_abs,2),size(x_abs,2));
W = rand([fbin,tidx]); %Initialize the noise stft
Alpha = zeros(fbin,tidx); % Initialize the weight A
b0 = 0.95; % Parameter Beta Vary from 15 to 30 in steps of 5
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
    W(:,i) = W(:,i-1)+b.*(Alpha(:,i)).*x_abs(:,i);
end
end

%% Maximum Likelihood Based Suppression Rule pp.234(pdf) Xestimated(f,k) = Gain(f,k)Noisy(f,k)
% where Gain(f,k) = 0.5+0.5*sqrt((p-1)/p) where p is the a-posteriori SNR
%    y_lps = W.*x_lps; y_estimated_af = sqrt(10.^y_lps).*exp(sqrt(-1)*angle(x_stft));
   y_estimated_af = W.*x_stft;
   s_estimated_af = d_stft(1:fbin,1:tidx) - y_estimated_af;
   s_estm = real(istft(s_estimated_af,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen));
   y_estm = real(istft(y_estimated_af,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen));
%%
L = min(length(d),length(s_estm));
newSig = d(1:L,:)-s_estm(1:L,:);