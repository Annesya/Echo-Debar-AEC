%% Randomly choose the data 
% n = randi([1 400],1);
n = 389;
testSet_d = table2array(DataAnnotationS2(2*n,1));
testSet_x = table2array(DataAnnotationS2(2*n-1,1));
addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\test_set\clean\');
[x,f] = audioread(testSet_x);
[d,f] = audioread(testSet_d);
% Set STFT parameters
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
Lt = min(size(d_abs,2),size(x_abs,2));
d_abs = d_abs(:,1:Lt);x_abs = x_abs(:,1:Lt); 
fbin = size(d_abs,1); tidx = size(d_abs,2);
W = ones([fbin,tidx]); %Initialize the noise stft
Alpha = zeros([fbin,tidx]); % Initialize the weight A
b0 = 1.0005; % Parameter Beta Vary from 15 to 30 in steps of 5
err = zeros(fbin,tidx); % Function Gamma
pastFrame = 20;
itr = 1; % Number of iteration
%% ADAPTIVE FILTER
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
% WIENER FILTERING
%s_wiener = wiener2(d_abs,[3 3],y_estimated_af);
s_wiener = d_abs-y_estimated_af;
input1 = s_wiener; %%%%%%%%%%%%%%% Change this line
numFeatures = size(input1,1); tind = size(input1,2); 
modelIn = input1;
% Prepare Model Input
testX = reshape(modelIn,size(modelIn,1),1,size(modelIn,2));
testX = squeeze(num2cell(testX,[1 2]));
testY = predict(aecNetFullyConnected,testX);
s_abs_prdt = cell2mat(testY);
s_abs_prdt = double(reshape(s_abs_prdt,fbin,tidx));
%% Compare STFT result
figure;
subplot(3,1,1);
t1 = 1:size(d_abs,2); f1 = 1:size(d_abs,1);
waterfall(t1,f1,d_abs);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,2);
t2 = 1:size(s_wiener,2); f2 = 1:size(s_wiener,1);
waterfall(t2,f2,s_wiener);colormap jet; colorbar; view(0,90);axis xy; axis tight;
subplot(3,1,3);
t3 = 1:size(s_abs_prdt,2); f3 = 1:size(s_abs_prdt,1);
waterfall(t3,f3,s_abs_prdt);colormap jet; colorbar; view(0,90);axis xy; axis tight;
%% Compare TD result
s_prdt = istft((s_abs_prdt.*exp(sqrt(-1)*phase_all(:,1:Lt))),f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
s_prdt = real(s_prdt);

