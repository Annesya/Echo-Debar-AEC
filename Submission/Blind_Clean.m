for i=1:500
    fprintf(['Count No: ',num2str(i),'\n']);
    n = i;
    testSet_d = table2array(BlindTestSet(2*n,1));
    testSet_x = table2array(BlindTestSet(2*n-1,1));
    addpath('C:\Users\UNUSUAL SOLUTIONS\AEC-Challenge\datasets\blind_test_set\clean\')
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
    s_wiener = d_abs-y_estimated_af;
    input1 = s_wiener; 
    numFeatures = size(input1,1); tind = size(input1,2); 
    modelIn = input1;
    % Prepare Model Input
    testX = reshape(modelIn,size(modelIn,1),1,size(modelIn,2));
    testX = squeeze(num2cell(testX,[1 2]));
    testY = predict(aecNetFullyConnected,testX);
    s_abs_prdt = cell2mat(testY);
    s_abs_prdt = double(reshape(s_abs_prdt,fbin,tidx));
    s_prdt = istft((s_abs_prdt.*exp(sqrt(-1)*phase_all(:,1:Lt))),f_new,'Window',hamming(winLen,'periodic'),'OverlapLength',overlap,'FFTLength',fftLen);
    s_prdt = real(s_prdt);
    audiowrite(testSet_d,s_prdt,f_new);
end