function [corr_ftrs, names] = corr_features_f(data)
%
% Outputs correlation features among columns of given data in the frequency
% domain
%
% usage
% supply eeg fft data to compute the correlation coefficients and 
% eigenvalues in the frequency domain

% Normalize data
eeg_norm = normalize(data, 2);

% Obtain correlation coefficients
corr_coefs = corr(eeg_norm');

% Initialize string array and upper triangle coefficients vector
names_coefs = strings([1,((size(corr_coefs, 1)^2)-size(corr_coefs, 1))/2]);
coef_ftrs = zeros([1,((size(corr_coefs, 1)^2)-size(corr_coefs, 1))/2]);

% Initialize counter
idx = 1;

% Keep the upper triangle and save names
for i=1:size(corr_coefs, 1)
   for  j=i+1:size(corr_coefs, 2)
       coef_ftrs(1,idx) = corr_coefs(i,j);
       names_coefs(1,idx) = sprintf('freq_corr_coef%d',idx);
       idx = idx + 1;
   end
end

% Obtain sorted eigenvalues
eig_corr = sort(eig(corr_coefs));

% Initialize string array and upper triangle coefficients vector
names_eigs = strings([1,length(eig_corr)]);
eigs_ftrs = zeros([1,length(eig_corr)]);

% Organize eigenvalues and names
for i=1:length(eig_corr)
    eigs_ftrs(1,i) = eig_corr(i);
    names_eigs(1,i) = sprintf('freq_corr_eig%d',i);
end

% Concatenate features and names
corr_ftrs = [coef_ftrs, eigs_ftrs];
names = [names_coefs, names_eigs];
end