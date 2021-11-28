% Epilepsy Prediction Project - CS534
% Created by: Daniela Chanci Arrubla

close all; clear; clc;

% Define parent folder
parentDir = 'G:\Epilepsy_Data\Prediction';
% Get subfolders
folders = dir(parentDir);
names = {folders.name};
% Get a logical vector that tells which is a directory
flags = [folders.isdir] & ~strcmp(names, '.') & ~strcmp(names, '..');
% Obtain patients folders
patients = names(flags);

% Iterate through patients
for i=1:length(patients)
    patient = char(patients(i))
    files = dir(fullfile(parentDir, patient));
    filenames = {files.name};
    fileflags = ~strcmp(filenames, '.') & ~strcmp(filenames, '..');
    signals = filenames(fileflags);
    flag = 1;
    for j=1:length(signals)
        matfile = char(signals(j));
        matfile_dir = fullfile(parentDir, patient, matfile);
        
        % Split file name
        parts_name = split(matfile,"_");
        
        % Assign label
        label = char(parts_name(3));
        if strcmp(label, 'interictal') == 1
            label_num = 0;
        elseif strcmp(label, 'preictal') == 1
            label_num = 1;
        else
            label_num = 2;
        end
        
        % Exclude test files
        if (label_num == 0 | label_num == 1)
            
            file_struct = load(matfile_dir);
            fn = fieldnames(file_struct);
            data_struct = file_struct.(fn{1});
            
            % Read complete signal and sampling frequency
            complete_data = data_struct.data;
            fs = round(data_struct.sampling_frequency);
            
            % Split signal in 30 seconds segments
            for s=1:20
                if s==1
                    data = complete_data(:, [1:fs*30]);
                elseif s==20
                    data = complete_data(:, [fs*30*(s-1)+1:end]);
                else
                    data = complete_data(:, [fs*30*(s-1)+1:fs*30*s]);
                end
                
                % Define  and size
                N = size(data,1);
                len = size(data,2);
                
                % Preprocess eeg
                data = preprocess_signal(data, fs);
                
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
                end
                
                % Fill matrices
                features = [features, label_num];
                train_features = [train_features; features];
                names_train = [names, 'label'];
                
                % Add to flag
                flag = flag + 1;
            end
        end
    end
    
    % Organize outputs
    table_train = array2table(train_features([2:end], :), 'VariableNames',names_train);
    
    % Create CSV files
    writetable(table_train, strcat('F:\Users\user\Desktop\EMORY\Classes\Fall_2021\CS_534\Project\Prediction\Extracted_Features_Preprocessing\',patient, '.csv'));
end