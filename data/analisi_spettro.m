%% Final ONLINE Analysis: Comparison between SSM, Real G, and N-Real Poles
clear; clc; close all;

% --- Design Parameters ---
rootFolder = 'harmonic_curves'; 
fs = 1 / 300;       % Sampling every 300s (5 min)
N_fft = 1024;       

% 1. 4th Order Butterworth Filter Configuration
fc = 1/5400;        % Cutoff at 1.5 hours
order_butt = 4;         
[b_butt, a_butt] = butter(order_butt, fc/(fs/2), 'low'); 

% 2. N-Real Poles Filter Configuration
N_poli_reali = 4;    % Number of poles (equal to Butterworth order for fair comparison)
p = exp(-2*pi*fc/fs); 

% Recursive calculation of the denominator (1 - p*z^-1)^N
den_Np = 1;
for k = 1:N_poli_reali
    den_Np = conv(den_Np, [1, -p]);
end
num_Np = (1-p)^N_poli_reali;  % Unit gain: (1-p)^N

% 3. Median Filter (Spike Killer)
medWin = 5; 

patients = dir(fullfile(rootFolder, 'patient*'));

%% --- Preliminary Calculation of the Reference Average (Target) ---
% Analyzing the spectral dynamics of all real signals to create a "Target" PSD
psd_ref_list = [];
for i = 1:length(patients)
    filePath = fullfile(rootFolder, patients(i).name, 'curves.mat');
    if exist(filePath, 'file')
        data = load(filePath);
        vars = fieldnames(data);
        for v = 1:length(vars)
            % Exclude 'i_hat' to create a clean reference target
            if ~strcmp(vars{v}, 'i_hat')
                sig = detrend(double(data.(vars{v})));
                [pxx, f] = periodogram(sig, [], N_fft, fs);
                psd_ref_list = [psd_ref_list, pxx];
            end
        end
    end
end
m_ref_db = 10*log10(mean(psd_ref_list, 2));

%% --- Main Loop per Patient ---
for i = 1:length(patients)
    patientID = patients(i).name;
    filePath = fullfile(rootFolder, patientID, 'curves.mat');
    
    if exist(filePath, 'file')
        data = load(filePath);
        
        if isfield(data, 'i_hat') && isfield(data, 'i_true')
            i_hat_raw = double(data.i_hat);
            i_true = double(data.i_true);
            
            % --- FILTER COMPARISON ---
            % A. N-Real Poles Filter (Normalized)
            i_hat_Np = filter(num_Np, den_Np, i_hat_raw);
            
            % B. Only 4th Order Butterworth
            i_hat_only_butt = filter(b_butt, a_butt, i_hat_raw);
            
            % C. Optimal Cascade (Median + Butterworth)
            sig_med = medfilt1(i_hat_raw, medWin);
            i_hat_ottimale = filter(b_butt, a_butt, sig_med);
            
            % PSD Calculation for Spectral Comparison
            [pxx_orig, ~] = periodogram(detrend(i_hat_raw), [], N_fft, fs);
            [pxx_Np, ~]   = periodogram(detrend(i_hat_Np), [], N_fft, fs);
            [pxx_butt, ~] = periodogram(detrend(i_hat_only_butt), [], N_fft, fs);
            [pxx_ott, ~]  = periodogram(detrend(i_hat_ottimale), [], N_fft, fs);
            
            % --- VISUALIZATION ---
            figure('Color', 'w', 'Name', ['Comparison - ' patientID], 'Position', [100 100 1200 850]);
            t = (0:length(i_hat_raw)-1) * 5 / 60; % Time in hours
            
            % SUBPLOT 1: TIME DOMAIN
            subplot(2,1,1);
            plot(t, i_hat_raw, 'Color', [0.7 0.3 0.3], 'LineWidth', 2, 'DisplayName', 'SSM prediction'); hold on;
            plot(t, i_true, 'k', 'LineWidth', 2, 'DisplayName', 'G real');
            plot(t, i_hat_Np, 'Color', [0.3 0.7 0.3], 'LineWidth', 2, 'DisplayName', [num2str(N_poli_reali), '-real poles']);
            
            % Commented out per user request:
            % plot(t, i_hat_only_butt, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Butterworth Only 4th');
            % plot(t, i_hat_ottimale, 'b', 'LineWidth', 2, 'DisplayName', 'MEDIANO + BUTTERWORTH');
            
            title(['Patient: ' patientID ' - Temporal Analysis']);
            xlabel('Time [hours]');
            ylabel('Amplitude'); 
            grid on; 
            legend('Location', 'best');
            
            % SUBPLOT 2: FREQUENCY DOMAIN
            subplot(2,1,2);
            plot(f, m_ref_db, 'Color', [0.3 0.3 0.7], 'LineWidth', 2, 'DisplayName', 'Avg spectral density realistic scenarios'); hold on;
            plot(f, 10*log10(pxx_orig),  'Color', [0.7 0.3 0.3],'LineWidth', 2, 'DisplayName', 'SSM prediction');
            plot(f, 10*log10(pxx_Np), 'Color', [0.3 0.7 0.3], 'LineWidth', 2, 'DisplayName', [ num2str(N_poli_reali), '-Real Poles']);
            % plot(f, 10*log10(pxx_ott), 'b', 'LineWidth', 2, 'DisplayName', 'Optimal PSD (Hybrid)');
            
            set(gca, 'XScale', 'log');
            title('Spectral Comparison: N-Real Poles vs Butterworth');
            xlabel('Frequency (Hz)'); 
            ylabel('Power (dB/Hz)'); 
            grid on;
            legend('Location', 'southwest');
        end
    end
end

%% Pole-Zero Comparison Analysis
figure('Name', 'Pole-Zero Map Comparison', 'Color', 'w');
subplot(1,2,1); pzmap(tf(num_Np, den_Np, 1/fs)); 
title([num2str(N_poli_reali), ' Overlapping Real Poles']);

subplot(1,2,2); pzmap(tf(b_butt, a_butt, 1/fs)); 
title(['Butterworth Order ', num2str(order_butt)]);