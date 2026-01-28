close all

use_abs_error = 0;

exp_id = [34,44,45,46,54,64,74];





color_101 = [0.85 0.4 0.3];        % Rosso pastello  -> modello 101
color_pv = [1 0.8 0.2];    % Giallo pastello  -> modelli lineari pavia
color_formula = [0.2 0.5 0.75];      % Blu pastello      -> formula
color_ssm = [0.3 0.75 0.4];      % Verde pastello    -> SSM


T = readtable('results_v3_peacks.xlsx');


%% impulse


p_i_ssm = read_exp(T, exp_id, 21:30, 'p_i_');

p_m_ssm = read_exp(T, exp_id, 31:40, 'p_m_');


p_i_ssm = str2double(p_i_ssm);
p_m_ssm = str2double(p_m_ssm);


%% delta I
figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
% Cambia il colore della terza barra a verde
h = bar([p_i_ssm']');

for i = 1:length(h)
    luminosita = 0.8 + 0.2 * (i - 4);  % Varia da 0.8 a più chiaro
    colore = color_ssm * luminosita;
    colore = min(colore, 1);  % Assicurati che non superi 1
    h(i).FaceColor = colore;
end

% Creare la legenda dinamicamente
legenda_base = {};

% Aggiungere gli SSM

for i = exp_id
    legenda_base{end+1} = ['ssm_' num2str(i)];
end

legend(legenda_base);


if use_abs_error
    ylim([0 1.5])
else
    ylim([-1.5 1.5])
end

% ----------------------------------

figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
E_i = [ ...
    p_i_ssm];


% Creare label dinamicamente
labels_base = {};
for i = exp_id
    labels_base{end+1} = ['ssm_' num2str(i)];
end

boxplot(E_i, ...
'Labels', labels_base, ...
'Whisker', 1.5);
grid on
ylabel('g(t+1) - g(t)')
title('peaks in only insulin scenario')

% Colori dei box
h = findobj(gca,'Tag','Box');
colors_base = {};

% Aggiungere i verdi per gli SSM
for i = 1:1+length(exp_id)
    luminosita = 0.8 + 0.2 * (i - 4);
    colore = color_ssm * luminosita;
    colore = min(colore, 1);
    colors_base{end+1} = colore;
end

% Applicare i colori
for i = 1:length(h)
    patch(get(h(i),'XData'), get(h(i),'YData'), ...
        colors_base{length(h)-i+1}, ...
    'FaceAlpha',0.6);
end
% 
% if use_abs_error
%     ylim([0 1.5])
% else
%     ylim([-1.5 1.5])
% end




%% M
figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
h = bar([ p_m_ssm']');

for i = 1:length(h)
    luminosita = 0.8 + 0.2 * (i - 4);  % Varia da 0.8 a più chiaro
    colore = color_ssm * luminosita;
    colore = min(colore, 1);  % Assicurati che non superi 1
    h(i).FaceColor = colore;
end
grid on
xlabel('patient')
ylabel('g(t+1) - g(t)')
title('peaks in only meal scenario')
% Creare la legenda dinamicamente
legenda_base = {};
% Aggiungere gli SSM
for i = exp_id
    legenda_base{end+1} = ['ssm_' num2str(i)];
end
legend(legenda_base);
if use_abs_error
    ylim([0 1.5])
else
    ylim([-1.5 1.5])
end

% ----------------------------------
figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
E_m = [ ...
    p_m_ssm];
% Creare label dinamicamente
labels_base = {};
for i = exp_id
    labels_base{end+1} = ['ssm_' num2str(i)];
end
boxplot(E_m, ...
'Labels', labels_base, ...
'Whisker', 1.5);
grid on
ylabel('g(t+1) - g(t)')
title('peaks in only meal scenario')
% Colori dei box
h = findobj(gca,'Tag','Box');
colors_base = {};
% Aggiungere i verdi per gli SSM
for i = 1:1+length(exp_id)
    luminosita = 0.8 + 0.2 * (i - 4);
    colore = color_ssm * luminosita;
    colore = min(colore, 1);
    colors_base{end+1} = colore;
end
% Applicare i colori
for i = 1:length(h)
    patch(get(h(i),'XData'), get(h(i),'YData'), ...
        colors_base{length(h)-i+1}, ...
    'FaceAlpha',0.6);
end






%% functions

function matrice = read_exp(T, numeri, righe, colum_start)

matrice = [];

for num = numeri
    nome_colonna = [colum_start num2str(num)];
    colonna = T{righe, nome_colonna};
    matrice = [matrice, colonna];  % Aggiungi come colonna
end

end