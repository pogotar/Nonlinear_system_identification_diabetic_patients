%% Analisi ONLINE Finale: Confronto Mediano, Butterworth e Media Mobile
clear; clc; close all;

% --- Parametri di Progetto ---
rootFolder = 'harmonic_curves'; 
fs = 1 / 300;       % 5 minuti



N_fft = 1024;       

% 1. Configurazione Filtro Butterworth Causale (Guadagno 1)

% Il Plateau: Quel "pavimento" orizzontale è rumore bianco/errore di stima. Iniziava circa a 3⋅10−4 Hz.
% La Scelta: Abbiamo impostato la frequenza di taglio a 2.0⋅10−4 Hz.
% 
%     In termini di periodo: T=1/f≈5000 secondi ≈ 1.4 ore.
% io modificato a 1 ora e mezza -> 90*60 = 5400 secondi

fc = 1/5400;        % Taglio identificato per eliminare il plateau
ordine = 4;         
[b, a] = butter(ordine, fc/(fs/2), 'low'); 

% 2. Filtro Mediano (Killer degli spike)
medWin = 5; 

% 3. Filtro Passabasso Semplice (Media Mobile - Moving Average)
% Usiamo una finestra di 5 campioni per confrontarla col mediano
movAvgWin = 4; 

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
            % A. Passabasso Semplice (Media Mobile)
            i_hat_movavg = movmean(i_hat_raw, movAvgWin);
            
            % B. Solo Butterworth (Lineare)
            i_hat_only_butt = filter(b, a, i_hat_raw);
            
            % C. Cascata Ottimale (Mediano + Butterworth)
            sig_med = medfilt1(i_hat_raw, medWin);
            i_hat_ottimale = filter(b, a, sig_med);
            
            % Calcolo PSD
            [pxx_orig, ~] = periodogram(detrend(i_hat_raw), [], N_fft, fs);
            [pxx_mov, ~] = periodogram(detrend(i_hat_movavg), [], N_fft, fs);
            [pxx_butt, ~] = periodogram(detrend(i_hat_only_butt), [], N_fft, fs);
            [pxx_ott, ~] = periodogram(detrend(i_hat_ottimale), [], N_fft, fs);
            
            % --- VISUALIZZAZIONE ---
            figure('Color', 'w', 'Name', ['Confronto Totale - ' patientID], 'Position', [100 100 1200 850]);
            t = (0:length(i_hat_raw)-1) * 5 / 60; 

            % SUBPLOT 1: TEMPO (Zoom sugli spike)
            subplot(2,1,1);
            plot(t, i_hat_raw, 'Color', [1 0.8 0.8], 'DisplayName', 'Originale (Spikes)'); hold on;
            plot(t, i_true, 'k', 'LineWidth', 2, 'DisplayName', 'i\_true (REALE)');
            plot(t, i_hat_movavg, 'c--', 'LineWidth', 1.5, 'DisplayName', 'Passabasso (Media Mobile)');
            plot(t, i_hat_only_butt, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Solo Butterworth');
            plot(t, i_hat_ottimale, 'b', 'LineWidth', 2, 'DisplayName', 'MEDIANO + BUTTERWORTH');
            title(['Paziente: ' patientID ' - Analisi Temporale']);
            ylabel('Ampiezza'); grid on; legend('Location', 'best');

            % SUBPLOT 2: FREQUENZA (Abbattimento rumore)
            subplot(2,1,2);
            plot(f, m_ref_db, 'Color', [0.3 0.3 0.3], 'LineWidth', 2, 'DisplayName', 'Target (Media Altri)'); hold on;
            plot(f, 10*log10(pxx_orig), 'r:', 'DisplayName', 'PSD i\_hat Originale');
            plot(f, 10*log10(pxx_mov), 'c--', 'DisplayName', 'PSD Media Mobile');
            plot(f, 10*log10(pxx_ott), 'b', 'LineWidth', 2, 'DisplayName', 'PSD Ottimale');
            
            set(gca, 'XScale', 'log');
            title('Confronto Spettrale: Perché la cascata è superiore?');
            xlabel('Frequenza (Hz)'); ylabel('Potenza (dB/Hz)'); grid on;
            legend('Location', 'southwest');
        end
    end
end


%% Analisi Poli-Zeri (Mappa del Filtro Sviluppato)
TF = tf(b, a, 1/fs); 
figure; pzmap(TF); title('Mappa Poli-Zeri del Filtro Online');