% Epilepsy Detection Project - CS534
% Created by: Daniela Chanci Arrubla

close all; clear; clc;

% Define parent folder
parentDir = 'G:\Epilepsy_Data\Detection';
% Get subfolders
folders = dir(parentDir);
names = {folders.name};
% Get a logical vector that tells which is a directory
flags = [folders.isdir] & ~strcmp(names, '.') & ~strcmp(names, '..');
% Obtain patients folders
patients = names(flags);

% Iterate through patients
for i=1:length(patients)
    patient = char(patients(i));
    files = dir(fullfile(parentDir, patient));
    filenames = {files.name};
    fileflags = ~strcmp(filenames, '.') & ~strcmp(filenames, '..');
    signals = filenames(fileflags);
    flag = 1;
    for j=1:length(signals)
        matfile = char(signals(j));
        matfile_dir = fullfile(parentDir, patient, matfile);
        load(matfile_dir);
        
        % Define sampling frequency and size
        fs = round(freq);
        N = size(data,1);
        len = size(data,2);
        
        % % Preprocess eeg
        % data = preprocess_signal(data, fs);
        
        % Obtain basic statistics in time
        [t_stat_ftrs, names_t_stat_ftrs]  = time_stats(data, N);
        
        % Obtain correlation features in time domain
        [t_corr_ftrs, names_t_corr_ftrs] = corr_features_t(data);
        
        % Apply Fourier Transform
        [eeg_fft, f] = fft_signal(data, fs, N, len);
        
        % Obtain power in bands
        [bands_ftrs, names_bands_ftrs] = bands_features(eeg_fft, f, N);
        
        % Obtain correlation features in frequency domain
        [f_corr_ftrs, names_f_corr_ftrs] = corr_features_f(eeg_fft);
        
        % Organize output table
        features = [t_stat_ftrs, t_corr_ftrs, bands_ftrs, f_corr_ftrs];
        names = [names_t_stat_ftrs, names_t_corr_ftrs, names_bands_ftrs, names_f_corr_ftrs];
        
        % Initialize matrix
        if flag == 1
            train_features = zeros(1, size(features,2)+1);
            test_features = zeros(size(features));
        end
        
        % Split file name
        parts_name = split(matfile,"_");
        
        % Assign label
        label = char(parts_name(3));
        if strcmp(label, 'interictal') == 1
            label_num = 0;
        elseif strcmp(label, 'ictal') == 1
            label_num = 1;
        else
            label_num = 2;
        end
        
        % Fill matrices
        if label_num == 2
            test_features = [test_features; features];
        else
            features = [features, label_num];
            train_features = [train_features; features];
            names_train = [names, 'label'];
        end
        
        % Add to flag
        flag = flag + 1;
    end
    
    % Organize outputs
    table_train = array2table(train_features([2:end], :), 'VariableNames',names_train);
    table_test = array2table(test_features([2:end], :), 'VariableNames',names);
    
    % Create CSV files
    writetable(table_train, strcat('F:\Users\user\Desktop\EMORY\Classes\Fall_2021\CS_534\Project\Detection\Extracted_Features\',patient, '_train.csv'));
    writetable(table_test, strcat('F:\Users\user\Desktop\EMORY\Classes\Fall_2021\CS_534\Project\Detection\Extracted_Features\',patient, '_test.csv'));
end