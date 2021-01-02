function [energy] = energyContour(d,winLength,overlapLength)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
F = 16000;
signal = d;
winLen = winLength;
overlap = overlapLength;
dataLen = size(signal,1);
winNum = round(dataLen/(winLen-overlap));
% bandpassFilt = designfilt('bandpassfir', 'StopbandFrequency1', 0.04, 'PassbandFrequency1', 0.05, 'PassbandFrequency2', 0.50, 'StopbandFrequency2', .62, 'StopbandAttenuation1', 60, 'PassbandRipple', 1, 'StopbandAttenuation2', 60);
% signal_filtered = filter(bandpassFilt,signal);
signal_filtered = signal;
% Find energy contour
for i=1:winNum
    startPt = max(1,(i-1)*winLen-overlap);
    endPt = min(dataLen,startPt+winLen-1);
    sig_chunk = signal_filtered(startPt:endPt,1);
    signal_win = sig_chunk.*hamming(length(winLen),'periodic');
    energy(i,1) = sum((signal_win).^2);
end
