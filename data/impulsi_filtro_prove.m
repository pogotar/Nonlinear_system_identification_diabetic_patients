%% Confronto Low-pass Filter vs Media Mobile vs Combinato
clear; close all; clc;

%% Parametri dei poli (cambia questi valori)
pole1 = 0.7;  % Primo polo
pole2 = 0.7;  % Secondo polo

%% Parametri filtro media mobile
N_avg = 2;  % Numero di campioni per la media

%% H1: Low-pass Filter
% H1(z) = G / (1 - a1*z^-1 - a2*z^-2)
a1_lp = pole1 + pole2;
a2_lp = -pole1 * pole2;
G_lp = 1;  % Guadagno

num_H1 = [G_lp];
den_H1 = [1, -a1_lp, -a2_lp];
H1 = tf(num_H1, den_H1, 1);

fprintf('===== H1: Low-pass Filter =====\n');
fprintf('Poli: %.4f, %.4f\n', pole1, pole2);
disp(H1);

%% H2: Filtro Media Mobile (Moving Average)
% H2(z) = (1 + z^-1 + z^-2 + ... + z^-N) / N
num_H2 = ones(1, N_avg) / N_avg;
den_H2 = 1;
H2 = tf(num_H2, den_H2, 1);

fprintf('\n===== H2: Media Mobile (N=%d) =====\n', N_avg);
disp(H2);

%% H3: Prodotto di H1 e H2 (in cascata)
H3 = H1 * H2;

fprintf('\n===== H3: Low-pass + Media Mobile (Combinato) =====\n');
disp(H3);

%% Plot Pole-Zero Map
figure('Position', [100 100 1600 600]);

subplot(1, 3, 1);
pzmap(H1);
title('H1: Low-pass (Poli)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; axis equal;
xlim([-1.5 1.5]); ylim([-1.5 1.5]);

subplot(1, 3, 2);
pzmap(H2);
title(sprintf('H2: Media Mobile (N=%d)', N_avg), 'FontSize', 12, 'FontWeight', 'bold');
grid on; axis equal;
xlim([-1.5 1.5]); ylim([-1.5 1.5]);

subplot(1, 3, 3);
pzmap(H3);
title('H3: H1 * H2 (Combinato)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; axis equal;
xlim([-1.5 1.5]); ylim([-1.5 1.5]);

%% Risposta in frequenza (Bode plot)
figure('Position', [100 750 1600 600]);

% Estraiamo i coefficienti da H3 (oggetto tf) per usarli con freqz
[num_H3, den_H3] = tfdata(H3, 'v'); 

% Frequenze normalizzate
[H1_resp, w] = freqz(num_H1, den_H1, 512);
[H2_resp, ~] = freqz(num_H2, den_H2, 512);
[H3_resp, ~] = freqz(num_H3, den_H3, 512); % Ora passiamo i coefficienti
% Converti in dB
mag_H1_dB = 20*log10(abs(H1_resp) + 1e-10);
mag_H2_dB = 20*log10(abs(H2_resp) + 1e-10);
mag_H3_dB = 20*log10(abs(H3_resp) + 1e-10);

subplot(2, 1, 1);
plot(w/pi, mag_H1_dB, 'b-', 'LineWidth', 2); hold on;
plot(w/pi, mag_H2_dB, 'g-', 'LineWidth', 2);
plot(w/pi, mag_H3_dB, 'r-', 'LineWidth', 2);
xlabel('Frequenza Normalizzata (×π rad/sample)', 'FontSize', 11);
ylabel('Magnitudo (dB)', 'FontSize', 11);
title('Risposta in Frequenza: Magnitudo', 'FontSize', 12, 'FontWeight', 'bold');
legend('H1: Low-pass', sprintf('H2: Media Mobile (N=%d)', N_avg), 'H3: Combinato', 'FontSize', 10);
grid on;

subplot(2, 1, 2);
phase_H1_deg = angle(H1_resp) * 180/pi;
phase_H2_deg = angle(H2_resp) * 180/pi;
phase_H3_deg = angle(H3_resp) * 180/pi;
plot(w/pi, phase_H1_deg, 'b-', 'LineWidth', 2); hold on;
plot(w/pi, phase_H2_deg, 'g-', 'LineWidth', 2);
plot(w/pi, phase_H3_deg, 'r-', 'LineWidth', 2);
xlabel('Frequenza Normalizzata (×π rad/sample)', 'FontSize', 11);
ylabel('Fase (gradi)', 'FontSize', 11);
title('Risposta in Frequenza: Fase', 'FontSize', 12, 'FontWeight', 'bold');
legend('H1: Low-pass', sprintf('H2: Media Mobile (N=%d)', N_avg), 'H3: Combinato', 'FontSize', 10);
grid on;

%% Risposta a impulso singolo
figure('Position', [100 1450 1600 600]);

% Impulso unitario
n = 0:70;
impulse_sig = [zeros(1, 20) 1 zeros(1, 50)];

% Filtra l'impulso
y_H1 = filter(num_H1, den_H1, impulse_sig);
y_H2 = filter(num_H2, den_H2, impulse_sig);

% H3: applica H1 poi H2 in cascata
y_H3 = filter(num_H1, den_H1, impulse_sig);
y_H3 = filter(num_H2, den_H2, y_H3);

subplot(1, 3, 1);
stem(n, y_H1, 'b', 'filled'); hold on;
stem(n, impulse_sig, 'k--', 'LineWidth', 1.5);
xlabel('Campione', 'FontSize', 11);
ylabel('Ampiezza', 'FontSize', 11);
title('H1: Low-pass - Impulso', 'FontSize', 12, 'FontWeight', 'bold');
legend('Output H1', 'Impulso', 'FontSize', 10);
grid on;

subplot(1, 3, 2);
stem(n, y_H2, 'g', 'filled'); hold on;
stem(n, impulse_sig, 'k--', 'LineWidth', 1.5);
xlabel('Campione', 'FontSize', 11);
ylabel('Ampiezza', 'FontSize', 11);
title(sprintf('H2: Media Mobile (N=%d) - Impulso', N_avg), 'FontSize', 12, 'FontWeight', 'bold');
legend('Output H2', 'Impulso', 'FontSize', 10);
grid on;

subplot(1, 3, 3);
stem(n, y_H3, 'r', 'filled'); hold on;
stem(n, impulse_sig, 'k--', 'LineWidth', 1.5);
xlabel('Campione', 'FontSize', 11);
ylabel('Ampiezza', 'FontSize', 11);
title('H3: H1 * H2 - Impulso', 'FontSize', 12, 'FontWeight', 'bold');
legend('Output H3', 'Impulso', 'FontSize', 10);
grid on;

fprintf('\n===== OSSERVAZIONI =====\n');
fprintf('H1: Impulso passa grande, coda lunga\n');
fprintf('H2: Impulso attenuato subito (diviso per N)\n');
fprintf('H3: Combinazione - attenuazione immediata + coda lunga\n');

