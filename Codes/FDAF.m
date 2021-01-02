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
%% FDAF
L = length(d);
echoCanceller = dsp.FrequencyDomainAdaptiveFilter('Length', 2048, ...
                    'StepSize', 0.025, ...
                    'InitialPower', 0.01, ...
                    'AveragingFactor', 0.98, ...
                    'Method', 'Unconstrained FDAF');
echoCanceller.BlockLength = L;
[y1,e] = echoCanceller(y,d);