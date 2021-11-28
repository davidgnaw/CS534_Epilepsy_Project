function [bands_ftrs, names] = bands_features(data, f, N)
%
% Outputs average power in selected frequency bands
%
% usage
% supply eeg fft data, frequency vector, and number of channels to compute
% the average frequency power in the selected frequency bands

% Define bands
bands = [[0.1, 4]; [4, 8]; [8, 15]; [15, 30]; [30, 90]; [90, 170]];

% Initialize string array and upper triangle coefficients vector
names = strings([1,size(bands,1)*N]);
bands_ftrs = zeros([1,size(bands,1)*N]);

% Initialize counter
idx = 1;

for b=1:size(bands,1)
    for i=1:N
        band_idxs = find(f(i,:)>bands(b,1) & f(i,:)<bands(b,2));
        bands_ftrs(1, idx) = mean(data(i, band_idxs));
        names(1,idx) = sprintf('power_band%d_ch%d',b,i);
        idx = idx + 1;
    end
end
end