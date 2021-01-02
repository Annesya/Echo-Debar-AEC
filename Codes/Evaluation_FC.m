n = randi([1 400],1);
testSet_d = table2array(DataAnnotationS3(2*n,1));
testSet_x = table2array(DataAnnotationS3(2*n-1,1));
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\test_set\noisy\');
[x,f] = audioread(testSet_x);
[d,f] = audioread(testSet_d);
f_new = 16000;
winLen = (16*10^-3)*f_new; % 16ms window
overlap = winLen/2; % 50% overlp
fftLen = winLen*2;
% Obtain the STFT
d_stft = stft(d,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
x_stft = stft(x,f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
phase_all = angle(d_stft); % Preserve the phase
d_abs = abs(d_stft);
x_abs = abs(x_stft);
d_lps = log10(((d_abs).^2));
x_lps = log10(((x_abs).^2));
Lt = min(size(d_lps,2),size(x_lps,2));
input1 = [x_lps(:,1:Lt)
        d_lps(:,1:Lt)];
numFeatures = size(input1,1); tind = size(input1,2); numSegments = 2;
input = [input1(:,1:numSegments - 1,:), input1];
stftSegments = zeros(numFeatures, numSegments , size(input,2) - numSegments + 1,size(input,3));
for num1 = 1:size(input,3)
    for index = 1:size(input,2) - numSegments + 1
     stftSegments(:,:,index,num1) = (input(:,index:index + numSegments - 1,num1));
    end
end
inpx = squeeze(num2cell(stftSegments,[1 2]));
outputY = predict(aecNetFullyConnected,inpx);
s_lps = cell2mat(outputY);
s_lps = reshape(s_lps,numFeatures/2,floor(length(s_lps)*2/numFeatures));
figure;
subplot(3,1,1);
t1 = 1:size(d_lps,2); f1 = 1:size(d_lps,1);
waterfall(t1,f1,d_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,2);
t2 = 1:size(x_lps,2); f2 = 1:size(x_lps,1);
waterfall(t2,f2,x_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,3);
t3 = 1:size(s_lps,2); f3 = 1:size(s_lps,1);
waterfall(t3,f3,s_lps);colormap jet; colorbar; view(0,90);axis xy; axis tight;

