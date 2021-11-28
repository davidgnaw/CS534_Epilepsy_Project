function filtered_eeg = preprocess_signal(signal, fs)
%
% Filters eeg data using cutoff frequencies (0.1Hz 180Hz)
%
% usage
% supply eeg data and sampling frequency to design 
% and apply Butterworth bandpass filter

% Define number of channels
N = size(signal,1);

% Preallocate filtered eeg signal
filtered_eeg = zeros(size(signal));

% Design Bandpass Filter
[b, a] = butter(2, [0.1 180]/(0.5*fs), 'bandpass');

% Filter signal per channel
for i=1:N
    filtered_eeg(i,:) = filtfilt(b, a, signal(i,:));
end
end