# Seizure Detection

[Extracted features link](https://drive.google.com/drive/folders/1CZAPvqD5LiW74SfSmzMEDoEuUTj0UIGs?usp=sharing)

## Preprocessing

- Butterworth bandpass filter between 0.1Hz and 180 Hz.

## Features

### Time Domain (Per Channel)

- Mean
- Std
- Kurtosis
- Skewness

### Time Domain (Global)

- Upper triangle correlation coefficients (Correlation matrix between channels)
- Sorted eigenvalues of the correlation matrix

### Frequency Domain (Per Channel)

- Average frequency power for the following bands: [[0.1, 4]; [4, 8]; [8, 15]; [15, 30]; [30, 90]; [90, 170]]

### Frequency Domain (Global)

- Upper triangle correlation coefficients (Correlation matrix between channels)
- Sorted eigenvalues of the correlation matrix