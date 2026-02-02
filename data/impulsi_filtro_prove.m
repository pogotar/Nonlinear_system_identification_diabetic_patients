%% Analisi Comparativa: Filtri IIR, FIR e Butterworth (Guadagno Unitario)
clear; close all; clc;

% --- Parametri di Progetto ---
n_samples = 80;                 % Lunghezza della simulazione
n = 0:n_samples-1;              
pole_val = 0.8;                 % Valore dei poli reali per H1 (più vicino a 1 = più smoothing)
N_avg = 4;                      % Finestra Media Mobile per H2
Wn_butt = 0.15;                 % Frequenza di taglio Butterworth (normalizzata 0-1)
order_butt = 4;                 % Ordine del Butterworth

% --- 1. Filtro H1: Due Poli Reali (Normalizzato per Guadagno 1) ---
% Formula: H(z) = (1-p)^2 / (1 - p*z^-1)^2
G_lp = (1 - pole_val)^2;
num_H1 = G_lp;
den_H1 = [1, -2*pole_val, pole_val^2];

% --- 2. Filtro H2: Media Mobile (Normalizzato per Guadagno 1) ---
num_H2 = ones(1, N_avg) / N_avg;
den_H2 = 1;

% --- 3. Filtro H3: Cascata H1 * H2 ---
% La convoluzione dei coefficienti equivale al prodotto delle FT
num_H3 = conv(num_H1, num_H2);
den_H3 = conv(den_H1, den_H2);

% --- 4. Filtro H4: Butterworth 4° Ordine ---
[b4, a4] = butter(order_butt, Wn_butt, 'low');

% Creazione oggetti Transfer Function per mappe poli-zeri
H1_tf = tf(num_H1, den_H1, 1);
H2_tf = tf(num_H2, den_H2, 1);
H3_tf = tf(num_H3, den_H3, 1);
H4_tf = tf(b4, a4, 1);

% ==========================================================
% FIGURE 1: Mappe Poli e Zeri
% ==========================================================
figure('Name', 'Mappe Poli-Zeri', 'Position', [100, 100, 1500, 400], 'Color', 'w');
tfs = {H1_tf, H2_tf, H3_tf, H4_tf};
titles = {'H1: 2 Poli Reali', ['H2: Media Mobile (N=',num2str(N_avg),')'], 'H3: Cascata H1*H2', 'H4: Butterworth 4° Ordine'};

for i = 1:4
    subplot(1, 4, i);
    pzmap(tfs{i});
    grid on; axis equal;
    title(titles{i}, 'FontSize', 10);
end

% ==========================================================
% FIGURE 2: Risposta in Frequenza (Bode)
% ==========================================================
figure('Name', 'Risposta in Frequenza', 'Position', [150, 150, 1200, 700], 'Color', 'w');
[h1_r, w] = freqz(num_H1, den_H1, 1024);
[h2_r, ~] = freqz(num_H2, den_H2, 1024);
[h3_r, ~] = freqz(num_H3, den_H3, 1024);
[h4_r, ~] = freqz(b4, a4, 1024);

subplot(2, 1, 1);
plot(w/pi, 20*log10(abs(h1_r)+1e-10), 'b', 'LineWidth', 1.5); hold on;
plot(w/pi, 20*log10(abs(h2_r)+1e-10), 'g', 'LineWidth', 1.5);
plot(w/pi, 20*log10(abs(h3_r)+1e-10), 'r', 'LineWidth', 2);
plot(w/pi, 20*log10(abs(h4_r)+1e-10), 'm--', 'LineWidth', 2);
grid on; ylabel('Magnitudo (dB)'); ylim([-60 5]);
title('Risposta in Ampiezza (Tutti partono da 0 dB)');
legend('H1 (Reali)', 'H2 (MA)', 'H3 (Cascata)', 'H4 (Butterworth)', 'Location', 'southwest');

subplot(2, 1, 2);
plot(w/pi, unwrap(angle(h1_r))*180/pi, 'b'); hold on;
plot(w/pi, unwrap(angle(h2_r))*180/pi, 'g');
plot(w/pi, unwrap(angle(h3_r))*180/pi, 'r', 'LineWidth', 2);
plot(w/pi, unwrap(angle(h4_r))*180/pi, 'm--', 'LineWidth', 2);
grid on; ylabel('Fase (gradi)'); xlabel('Frequenza Normalizzata (\times\pi rad/sample)');
title('Risposta in Fase');

% ==========================================================
% FIGURE 3: Risposta all'Impulso con Confronto Originale
% ==========================================================
figure('Name', 'Risposta Temporale all''Impulso', 'Position', [200, 200, 1500, 450], 'Color', 'w');
impulse_sig = [1, zeros(1, n_samples-1)]; % Impulso unitario a t=0

y_results = {filter(num_H1, den_H1, impulse_sig), ...
             filter(num_H2, den_H2, impulse_sig), ...
             filter(num_H3, den_H3, impulse_sig), ...
             filter(b4, a4, impulse_sig)};
colors = {'b', 'g', 'r', 'm'};

for i = 1:4
    subplot(1, 4, i);
    % Disegna l'impulso originale come riferimento (nero tratteggiato)
    stem(n(1:40), impulse_sig(1:40), 'k--', 'LineWidth', 1, 'Marker', 'none'); hold on;
    % Disegna l'uscita del filtro
    stem(n(1:40), y_results{i}(1:40), colors{i}, 'filled', 'LineWidth', 1.5);
    
    grid on; title(titles{i});
    xlabel('Campioni (n)'); ylabel('Ampiezza');
    legend('Impulso Originale', 'Segnale Filtrato', 'FontSize', 8);
    % Imposta limiti fissi per facilitare il confronto visivo
    ylim([-0.1 1.1]); 
end

fprintf('Analisi terminata. Tutti i filtri hanno guadagno unitario in DC.\n');