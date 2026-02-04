%% Analisi Spettrale Globale e Confronto Filtraggio Ottimale
clear; clc; close all;

% --- 1. Parametri di Progetto ---
rootFolder = 'harmonic_curves'; 
fs = 1 / 300;       % Campionamento ogni 5 minuti (1/300 Hz)
N_fft = 1024;       
fc = 1/5400;        % Taglio a 1.5 ore (identificato per abbattere il noise plateau)

% Configurazione Filtri
ordine_butt = 4;    % Ordine Butterworth
N_poli_reali = 4;   % Ordine N-Poli Reali
medWin = 5;         % Finestra filtro mediano (per gli spike)

% Progetto Filtro Butterworth
[b_butt, a_butt] = butter(ordine_butt, fc/(fs/2), 'low'); 

% Progetto Filtro N-Poli Reali (Normalizzato)
p = exp(-2*pi*fc/fs); 
den_Np = 1;
for k = 1:N_poli_reali
    den_Np = conv(den_Np, [1, -p]);
end
num_Np = (1-p)^N_poli_reali; 

% Recupero lista pazienti
patients = dir(fullfile(rootFolder, 'patient*'));

%% --- 2. Analisi Statistica: Calcolo PSD di Riferimento (Target) ---
% Scopo: Capire qual è la dinamica "giusta" analizzando tutti i segnali reali
% tranne quello che vogliamo stimare (i_hat).
psd_ref_list = [];

fprintf('Analisi spettrale di riferimento in corso...\n');
for i = 1:length(patients)
    filePath = fullfile(rootFolder, patients(i).name, 'curves.mat');
    if exist(filePath, 'file')
        data = load(filePath);
        vars = fieldnames(data);
        for v = 1:length(vars)
            % Escludiamo 'i_hat' (la stima sporca) per avere il riferimento pulito
            if ~strcmp(vars{v}, 'i_hat')
                sig = detrend(double(data.(vars{v})));
                [pxx, f] = periodogram(sig, [], N_fft, fs);
                psd_ref_list = [psd_ref_list, pxx];
            end
        end
    end
end

% Media spettrale target (in dB)
m_ref_db = 10*log10(mean(psd_ref_list, 2));
fprintf('Target spettrale calcolato correttamente.\n');

%% --- 3. Loop Principale: Filtraggio e Validazione ---
for i = 1:length(patients)
    patientID = patients(i).name;
    filePath = fullfile(rootFolder, patientID, 'curves.mat');
    
    if exist(filePath, 'file')
        data = load(filePath);
        
        if isfield(data, 'i_hat') && isfield(data, 'i_true')
            i_hat_raw = double(data.i_hat);
            i_true = double(data.i_true);
            
            % --- ESECUZIONE FILTRAGGIO ---
            % A. Filtro N-Poli Reali (Morbidezza massima)
            i_hat_Np = filter(num_Np, den_Np, i_hat_raw);
            
            % B. Solo Butterworth (Selettività frequenza)
            i_hat_butt = filter(b_butt, a_butt, i_hat_raw);
            
            % C. CASCATA OTTIMALE: Mediano (rimuove spike) + Butterworth (liscia)
            sig_med = medfilt1(i_hat_raw, medWin);
            i_hat_ottimale = filter(b_butt, a_butt, sig_med);
            
            % Calcolo PSD individuali
            [pxx_orig, ~] = periodogram(detrend(i_hat_raw), [], N_fft, fs);
            [pxx_Np, ~]   = periodogram(detrend(i_hat_Np), [], N_fft, fs);
            [pxx_ott, ~]  = periodogram(detrend(i_hat_ottimale), [], N_fft, fs);
            
            % --- VISUALIZZAZIONE ---
            figure('Color', 'w', 'Name', ['Analisi - ' patientID], 'Position', [100 100 1200 850]);
            t = (0:length(i_hat_raw)-1) * 5 / 60; % Ore
            
            % Subplot 1: Tempo (Dinamica del segnale)
            subplot(2,1,1);
            plot(t, i_hat_raw, 'Color', [1 0.7 0.7], 'DisplayName', 'Grezzo (Spikes)'); hold on;
            plot(t, i_true, 'k', 'LineWidth', 2, 'DisplayName', 'Target Reale');
            plot(t, i_hat_Np, 'g--', 'LineWidth', 1.5, 'DisplayName', [num2str(N_poli_reali), '-Poli Reali']);
            plot(t, i_hat_ottimale, 'b', 'LineWidth', 2, 'DisplayName', 'MEDIANO + BUTTERWORTH');
            title(['Analisi Temporale: ' patientID]);
            ylabel('Ampiezza'); grid on; legend('Location', 'best');
            
            % Subplot 2: Frequenza (Validazione Spettrale)
            subplot(2,1,2);
            plot(f, m_ref_db, 'Color', [0.3 0.3 0.3], 'LineWidth', 2, 'DisplayName', 'Target Spettrale (Media Altri)'); hold on;
            plot(f, 10*log10(pxx_orig), 'r:', 'DisplayName', 'PSD Grezza');
            plot(f, 10*log10(pxx_Np), 'g--', 'DisplayName', 'PSD N-Poli');
            plot(f, 10*log10(pxx_ott), 'b', 'LineWidth', 2, 'DisplayName', 'PSD Cascata Ottimale');
            
            set(gca, 'XScale', 'log');
            title('Confronto Spettrale: Il filtro segue la dinamica naturale?');
            xlabel('Frequenza (Hz)'); ylabel('Potenza (dB/Hz)'); grid on;
            legend('Location', 'southwest');
        end
    end
end