function [eeg_stats, names] = time_stats(data, channels)
%
% Outputs basic statistic features in the time domain
%
% usage
% supply filtered eeg data and number of channels to compute the time
% domain statistics

% Define stats
stats_names = ["mean", "std", "kurtosis", "skewness"];

% Initialize string array and stats matrix
names = strings([1,channels*length(stats_names)]);
eeg_stats = zeros(1,channels*length(stats_names));
stats_matrix = zeros(channels,length(stats_names));

% Initialize counter
idx = 1;

% Compute stats
stats_matrix(:,1) = mean(data,2);
stats_matrix(:,2) = std(data,0,2);
stats_matrix(:,3) = kurtosis(data,1,2);
stats_matrix(:,4) = skewness(data,1,2);

% Organize output table
for i=1:size(stats_matrix, 2)
   for j=1:size(stats_matrix, 1)  
       eeg_stats(1,idx) = stats_matrix(j,i);
       names(1,idx) = strcat(stats_names(i),sprintf('_ch%d',j));
       idx = idx + 1;
   end
end
end
