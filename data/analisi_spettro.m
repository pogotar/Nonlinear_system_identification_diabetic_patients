%% Analisi ONLINE Finale: Confronto Mediano, Butterworth e N-Poli Reali
clear; clc; close all;

% --- Parametri di Progetto ---
rootFolder = 'harmonic_curves'; 
fs = 1 / 300;       % Campionamento ogni 300s (5 min)
N_fft = 1024;       

% 1. Configurazione Filtro Butterworth 4° Ordine
fc = 1/5400;        % Taglio a 1.5 ore
ordine_butt = 4;         
[b_butt, a_butt] = butter(ordine_butt, fc/(fs/2), 'low'); 

% 2. Configurazione Filtro a N-Poli Reali (Sostituisce il 2-poli)
N_poli_reali = 4;    % Numero di poli (uguale all'ordine Butterworth per confronto equo)
p = exp(-2*pi*fc/fs); 

% Calcolo ricorsivo del denominatore (1 - p*z^-1)^N
den_Np = 1;
for k = 1:N_poli_reali
    den_Np = conv(den_Np, [1, -p]);
end
num_Np = (1-p)^N_poli_reali;  % Guadagno unitario: (1-p)^N

% 3. Filtro Mediano (Killer degli spike)
medWin = 5; 

patients = dir(fullfile(rootFolder, 'patient*'));

%% --- Calcolo Preliminare della Media di Riferimento ---
psd_ref_list = [];
for i = 1:length(patients)
    filePath = fullfile(rootFolder, patients(i).name, 'curves.mat');
    if exist(filePath, 'file')
        data = load(filePath);
        vars = fieldnames(data);
        for v = 1:length(vars)
            if ~strcmp(vars{v}, 'i_hat')
                sig = detrend(double(data.(vars{v})));
                [pxx, f] = periodogram(sig, [], N_fft, fs);
                psd_ref_list = [psd_ref_list, pxx];
            end
        end
    end
end
m_ref_db = 10*log10(mean(psd_ref_list, 2));

%% --- Loop Principale per Paziente ---
for i = 1:length(patients)
    patientID = patients(i).name;
    filePath = fullfile(rootFolder, patientID, 'curves.mat');
    
    if exist(filePath, 'file')
        data = load(filePath);
        
        if isfield(data, 'i_hat') && isfield(data, 'i_true')
            i_hat_raw = double(data.i_hat);
            i_true = double(data.i_true);
            
            % --- CONFRONTO FILTRI ---
            % A. Filtro N-Poli Reali (Normalizzato)
            i_hat_Np = filter(num_Np, den_Np, i_hat_raw);
            
            % B. Solo Butterworth 4° Ordine
            i_hat_only_butt = filter(b_butt, a_butt, i_hat_raw);
            
            % C. Cascata Ottimale (Mediano + Butterworth)
            sig_med = medfilt1(i_hat_raw, medWin);
            i_hat_ottimale = filter(b_butt, a_butt, sig_med);
            
            % Calcolo PSD per confronto spettrale
            [pxx_orig, ~] = periodogram(detrend(i_hat_raw), [], N_fft, fs);
            [pxx_Np, ~]   = periodogram(detrend(i_hat_Np), [], N_fft, fs);
            [pxx_butt, ~] = periodogram(detrend(i_hat_only_butt), [], N_fft, fs);
            [pxx_ott, ~]  = periodogram(detrend(i_hat_ottimale), [], N_fft, fs);
            
            % --- VISUALIZZAZIONE ---
            figure('Color', 'w', 'Name', ['Confronto - ' patientID], 'Position', [100 100 1200 850]);
            t = (0:length(i_hat_raw)-1) * 5 / 60; % Tempo in ore
            
            % SUBPLOT 1: TEMPO
            subplot(2,1,1);
            plot(t, i_hat_raw, 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', 'Originale (Spikes)'); hold on;
            plot(t, i_true, 'k', 'LineWidth', 2, 'DisplayName', 'i\_true (REALE)');
            plot(t, i_hat_Np, 'g--', 'LineWidth', 2, 'DisplayName', [num2str(N_poli_reali), '-Poli Reali']);
            plot(t, i_hat_only_butt, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Solo Butterworth 4°');
            plot(t, i_hat_ottimale, 'b', 'LineWidth', 2, 'DisplayName', 'MEDIANO + BUTTERWORTH');
            title(['Paziente: ' patientID ' - Analisi Temporale']);
            ylabel('Ampiezza'); grid on; legend('Location', 'best');
            
            % SUBPLOT 2: FREQUENZA
            subplot(2,1,2);
            plot(f, m_ref_db, 'Color', [0.3 0.3 0.3], 'LineWidth', 2, 'DisplayName', 'Target'); hold on;
            plot(f, 10*log10(pxx_orig), 'r:', 'DisplayName', 'PSD Originale');
            plot(f, 10*log10(pxx_Np), 'g--', 'LineWidth', 2, 'DisplayName', ['PSD ', num2str(N_poli_reali), '-Poli Reali']);
            plot(f, 10*log10(pxx_ott), 'b', 'LineWidth', 2, 'DisplayName', 'PSD Ottimale');
            
            set(gca, 'XScale', 'log');
            title('Confronto Spettrale: N-Poli Reali vs Butterworth');
            xlabel('Frequenza (Hz)'); ylabel('Potenza (dB/Hz)'); grid on;
            legend('Location', 'southwest');
        end
    end
end

%% Analisi Poli-Zeri di Confronto
figure('Name', 'Mappa Poli-Zeri Filtri Finali', 'Color', 'w');
subplot(1,2,1); pzmap(tf(num_Np, den_Np, 1/fs)); 
title([num2str(N_poli_reali), ' Poli Reali Sovrapposti']);
subplot(1,2,2); pzmap(tf(b_butt, a_butt, 1/fs)); 
title(['Butterworth ', num2str(ordine_butt), '° Ordine']);