modelOut = predict(aecNetFullyConnected,modelIn);
enhancedS = cell2mat(modelOut);
enhancedS = (reshape(enhancedS,size(s_estimated_af,1),size(s_estimated_af,2))).*d_abs;
enhancedS_com = enhancedS.*exp(sqrt(-1)*newSig_phase);
estm_rnn = real(istft(enhancedS_com,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen));
%%
figure;
subplot(2,1,1)
stft(d,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);
subplot(2,1,2)
stft(y_estm,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);ylim([0 Inf]);
%%
tr = 1:tidx; fr = 1:fbin;
figure;
subplot(2,1,1)
waterfall(tr,fr,input1); view(0,90);colorbar;
subplot(2,1,2)
waterfall(tr,fr,s_est_abs); view(0,90);colorbar;

