figure;
subplot(3,1,1);
t1 = 1:size(d_lps,2); f1 = 1:size(d_lps,1);
waterfall(t1,f1,d_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,2);
t2 = 1:size(x_lps,2); f2 = 1:size(x_lps,1);
waterfall(t2,f2,x_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,3);
t3 = 1:size(input,2); f3 = 1:size(input,1);
waterfall(t3,f3,input);colormap jet; colorbar; view(0,90);axis xy; axis tight;
%%
figure;
subplot(4,1,1);
t1 = 1:size(d_abs,2); f1 = 1:size(d_abs,1);
waterfall(t1,f1,d_abs);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(4,1,2);
t2 = 1:size(x_abs,2); f2 = 1:size(x_abs,1);
waterfall(t2,f2,x_abs);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(4,1,3);
% t3 = 1:size(input,2); f3 = 1:size(input,1);
% waterfall(t3,f3,input);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(4,1,4);
t4 = 1:size(y_estimated_af,2); f4 = 1:size(y_estimated_af,1);
waterfall(t4,f4,y_estimated_af);colormap jet; colorbar; view(0,90);axis xy; axis tight;zlim([0 25]);
%%
s_wiener = wiener2(d_abs,[3 3],y_estimated_af);
s1 = s_wiener;
s1fft = s1.*exp(sqrt(-1)*phase_all(:,1:size(s1,2)));
s1_t = istft(s1fft,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
s_Estm_wiener = real(s1_t);
frange = 1:fbin;trange = 1:tidx;
figure;
waterfall(trange,frange,s_estimated_af);colormap jet; colorbar; view(0,90);axis xy; axis tight;
%%
[esTSNR,esHRNR]=WienerNoiseReduction(y_Estm,f_new,800);
%%
y_stft = stft(y,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
y_abs = abs(y_stft);
Worg = y_abs./x_abs;
%%
x1 = sqrt(10.^input);
x1fft = x1.*exp(sqrt(-1)*phase_all(:,1:size(x1,2)));
x1_t = istft(x1fft,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
x1_t = real(x1_t);
%%
y1 = y_estimated_af;
y1fft = y1.*exp(sqrt(-1)*phase_all(:,1:size(y1,2)));
y1_t = istft(y1fft,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
y1_t = real(y1_t);
%%
s1 = s_estimated_af;
s1fft = s1.*exp(sqrt(-1)*phase_all(:,1:size(s1,2)));
s1_t = istft(s1fft,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
s1_t = real(s1_t);