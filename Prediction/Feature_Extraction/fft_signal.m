function [fft_eeg, f] = fft_signal(signal, fs, channels, len)
%
% Computes fft of signal
%
% usage
% supply preprocessed eeg data, number of channels, the frequency, and the
% datapoints length to obtain the fft of the signal

% Preallocate fft and f
fft_eeg = zeros(channels,(len/2)+1);
f = zeros(channels,(len/2)+1);

% Apply fft per channel
for i=1:channels
    freq_signal = fft(signal(i,:));
    P2 = abs(freq_signal/len);
    P1 = P2(1:len/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    fft_eeg(i,:) = P1;
    f(i,:) = fs*(0:(len/2))/len;
end
end