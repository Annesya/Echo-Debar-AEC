%%
winLength = 0.05*f_new;
overlapLength = winLength/2;
[energy] = energyContour(d,winLength,overlapLength);
thrsE = max(energy)/5;
E_thrs = thresoldValue(energy,thrsE);
f0 = pitch(d,f_new,'Method','SRH','WindowLength',winLength,'OverlapLength',overlapLength);
thrsF = max(160,max(f0)/1.5);
F_thrs = thresoldValue(f0,thrsF);
%%
t_idx = 1:tind;
f_idx = 1:numFeatures;
figure;
subplot(3,1,1);
waterfall(t_idx,f_idx,d_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,2);
plot(E_thrs);xlim([0 1250]);
subplot(3,1,3);
waterfall(t_idx,f_idx,s_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
%%
figure;
subplot(3,1,1);
plot(E_thrs);
subplot(3,1,2);
plot(F_thrs);
subplot(3,1,3)
plot(E_thrs(1:399,1).*F_thrs);
%%
t_idx = 1:size(d_lps,2); f_idx = 1:size(d_lps,1);
subplot(3,1,1);
waterfall(t_idx,f_idx,d_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,2);
waterfall(t_idx,f_idx,x_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,3);
waterfall(t_idx,f_idx,s_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;