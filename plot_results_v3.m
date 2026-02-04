close all

use_abs_error = 0;

exp_id = [34,64,74,84,65,75];





color_101 = [0.85 0.4 0.3];        % Rosso pastello  -> modello 101
color_pv = [1 0.8 0.2];    % Giallo pastello  -> modelli lineari pavia
color_formula = [0.2 0.5 0.75];      % Blu pastello      -> formula
color_ssm = [0.3 0.75 0.4];      % Verde pastello    -> SSM


T = readtable('results_v3.xlsx');


%% impulse

delta_G_i_empiric = T.delta_G_empiric(21:30);
delta_G_i_formula = T.delta_G_formula(21:30);
delta_G_i_pv = T.delta_G_PV_personalized(21:30);
delta_G_i_101 = T.delta_G_101(21:30);
delta_G_i_ssm = read_exp(T, exp_id, 21:30, 'delta_G_SSM_');

delta_t_i_empiric = T.delta_t_empiric(21:30);
delta_t_i_pv = T.delta_t_PV_personalized(21:30);
delta_t_i_101 = T.delta_t_101(21:30);
delta_t_i_ssm = read_exp(T, exp_id, 21:30, 'delta_t_SSM_');


delta_G_m_empiric = T.delta_G_empiric(31:40);
delta_G_m_formula = T.delta_G_formula(31:40);
delta_G_m_pv = T.delta_G_PV_personalized(31:40);
delta_G_m_101 = T.delta_G_101(31:40);
delta_G_m_ssm = read_exp(T, exp_id, 31:40, 'delta_G_SSM_');

delta_t_m_empiric = T.delta_t_empiric(31:40);
delta_t_m_pv = T.delta_t_PV_personalized(31:40);
delta_t_m_101 = T.delta_t_101(31:40);
delta_t_m_ssm = read_exp(T, exp_id, 31:40, 'delta_t_SSM_');


if use_abs_error
    e_i_formula = abs(-delta_G_i_empiric + delta_G_i_formula)./delta_G_i_empiric;
    e_i_pv = abs(-delta_G_i_empiric + delta_G_i_pv)./delta_G_i_empiric;
    e_i_101 = abs(-delta_G_i_empiric + delta_G_i_101)./delta_G_i_empiric;
    e_i_ssm = abs(-delta_G_i_empiric + delta_G_i_ssm)./delta_G_i_empiric;

    e_m_formula = abs(-delta_G_m_empiric + delta_G_m_formula)./delta_G_m_empiric;
    e_m_pv = abs(-delta_G_m_empiric + delta_G_m_pv)./delta_G_m_empiric;
    e_m_101 = abs(-delta_G_m_empiric + delta_G_m_101)./delta_G_m_empiric;
    e_m_ssm = abs(-delta_G_m_empiric + delta_G_m_ssm)./delta_G_m_empiric;
else

    e_i_formula = (-delta_G_i_empiric + delta_G_i_formula)./delta_G_i_empiric;
    e_i_pv = (-delta_G_i_empiric + delta_G_i_pv)./delta_G_i_empiric;
    e_i_101 = (-delta_G_i_empiric + delta_G_i_101)./delta_G_i_empiric;
    e_i_ssm = (-delta_G_i_empiric + delta_G_i_ssm)./delta_G_i_empiric;

    t_i_pv = ( delta_t_i_pv)./delta_t_i_empiric -1;
    t_i_101 = ( delta_t_i_101)./delta_t_i_empiric-1;
    t_i_ssm = ( delta_t_i_ssm)./delta_t_i_empiric-1;


    e_m_formula = (-delta_G_m_empiric + delta_G_m_formula)./delta_G_m_empiric;
    e_m_pv = (-delta_G_m_empiric + delta_G_m_pv)./delta_G_m_empiric;
    e_m_101 = (-delta_G_m_empiric + delta_G_m_101)./delta_G_m_empiric;
    e_m_ssm = (-delta_G_m_empiric + delta_G_m_ssm)./delta_G_m_empiric;

    t_m_pv = ( delta_t_m_pv)./delta_t_m_empiric-1;
    t_m_101 = ( delta_t_m_101)./delta_t_m_empiric-1;
    t_m_ssm = ( delta_t_m_ssm)./delta_t_m_empiric-1;

end

%% delta I
figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
% Cambia il colore della terza barra a verde
h = bar([e_i_formula'; e_i_pv'; e_i_101'; e_i_ssm']');
h(1).FaceColor = color_formula; 
h(2).FaceColor = color_pv; 
h(3).FaceColor = color_101;

for i = 4:length(h)
    luminosita = 0.8 + 0.2 * (i - 4);  % Varia da 0.8 a più chiaro
    colore = color_ssm * luminosita;
    colore = min(colore, 1);  % Assicurati che non superi 1
    h(i).FaceColor = colore;
end

% Creare la legenda dinamicamente
legenda_base = {'prior', 'model pv', '101'};

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
    e_i_formula, ...
    e_i_pv, ...
    e_i_101, ...
    e_i_ssm];


% Creare label dinamicamente
labels_base = {'prior', 'model pv', '101'};
for i = exp_id
    labels_base{end+1} = ['ssm_' num2str(i)];
end

boxplot(E_i, ...
'Labels', labels_base, ...
'Whisker', 1.5);
grid on
ylabel('error')
title('normalized error in delta glycemia with just insulin')

% Colori dei box
h = findobj(gca,'Tag','Box');
colors_base = {color_formula, color_pv, color_101};

% Aggiungere i verdi per gli SSM
for i = 3:3+length(exp_id)
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

if use_abs_error
    ylim([0 1.5])
else
    ylim([-1.5 1.5])
end

%% delta I t

figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
% Cambia il colore della terza barra a verde
h = bar([ t_i_pv'; t_i_101'; t_i_ssm']');
h(1).FaceColor = color_pv; 
h(2).FaceColor = color_101;

for i = 3:length(h)
    luminosita = 0.8 + 0.2 * (i - 4);  % Varia da 0.8 a più chiaro
    colore = color_ssm * luminosita;
    colore = min(colore, 1);  % Assicurati che non superi 1
    h(i).FaceColor = colore;
end

% Creare la legenda dinamicamente
legenda_base = {'model pv', '101'};

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
    t_i_pv, ...
    t_i_101, ...
    t_i_ssm];


% Creare label dinamicamente
labels_base = { 'model pv', '101'};
for i = exp_id
    labels_base{end+1} = ['ssm_' num2str(i)];
end

boxplot(E_i, ...
'Labels', labels_base, ...
'Whisker', 1.5);
grid on
ylabel('error')
title('normalized error in delta t with just insulin')

% Colori dei box
h = findobj(gca,'Tag','Box');
colors_base = { color_pv, color_101};

% Aggiungere i verdi per gli SSM
for i = 2:2+length(exp_id)
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

if use_abs_error
    ylim([0 1.5])
else
    ylim([-1.5 1.5])
end



%% M
figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
h = bar([e_m_formula'; e_m_pv'; e_m_101'; e_m_ssm']');
h(1).FaceColor = color_formula;
h(2).FaceColor = color_pv;
h(3).FaceColor = color_101;
for i = 4:length(h)
    luminosita = 0.8 + 0.4 * (i - 4);  % Varia da 0.8 a più chiaro
    colore = color_ssm * luminosita;
    colore = min(colore, 1);  % Assicurati che non superi 1
    h(i).FaceColor = colore;
end
grid on
xlabel('patient')
ylabel('error')
title('normalized error in delta glycemia with just meal')
% Creare la legenda dinamicamente
legenda_base = {'prior', 'model pv', '101'};
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
    e_m_formula, ...
    e_m_pv, ...
    e_m_101, ...
    e_m_ssm];
% Creare label dinamicamente
labels_base = {'prior', 'model pv', '101'};
for i = exp_id
    labels_base{end+1} = ['ssm_' num2str(i)];
end
boxplot(E_m, ...
'Labels', labels_base, ...
'Whisker', 1.5);
grid on
ylabel('error')
title('normalized error in delta glycemia with just meal')
% Colori dei box
h = findobj(gca,'Tag','Box');
colors_base = {color_formula, color_pv, color_101};
% Aggiungere i verdi per gli SSM
for i = 3:3+length(exp_id)
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
if use_abs_error
    ylim([0 1.5])
else
    ylim([-1.5 1.5])
end


%% delta M t

figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
% Cambia il colore della terza barra a verde
h = bar([ t_m_pv'; t_m_101'; t_m_ssm']');
h(1).FaceColor = color_pv; 
h(2).FaceColor = color_101;

for i = 3:length(h)
    luminosita = 0.8 + 0.2 * (i - 4);  % Varia da 0.8 a più chiaro
    colore = color_ssm * luminosita;
    colore = min(colore, 1);  % Assicurati che non superi 1
    h(i).FaceColor = colore;
end

% Creare la legenda dinamicamente
legenda_base = {'model pv', '101'};

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
    t_m_pv, ...
    t_m_101, ...
    t_m_ssm];


% Creare label dinamicamente
labels_base = { 'model pv', '101'};
for i = exp_id
    labels_base{end+1} = ['ssm_' num2str(i)];
end

boxplot(E_i, ...
'Labels', labels_base, ...
'Whisker', 1.5);
grid on
ylabel('error')
title('normalized error in delta t with just meal')

% Colori dei box
h = findobj(gca,'Tag','Box');
colors_base = { color_pv, color_101};

% Aggiungere i verdi per gli SSM
for i = 2:2+length(exp_id)
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

if use_abs_error
    ylim([0 1.5])
else
    ylim([-1.5 1.5])
end


%%  FIT



FIT_train_linear_PV = T.FIT_PV_personalized(1:10);
FIT_test_linear_PV = T.FIT_PV_personalized(11:20);
FIT_train_101 = T.FIT_101(1:10);
FIT_test_101 = T.FIT_101(11:20);
FIT_train_SSM = read_exp(T, exp_id, 1:10, 'FIT_SSM_');
FIT_test_SSM = read_exp(T, exp_id, 11:20, 'FIT_SSM_');

figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
h = bar([FIT_train_linear_PV'; FIT_test_linear_PV'; FIT_train_SSM'; FIT_test_SSM']');
h(1).FaceColor = color_pv;
h(2).FaceColor = color_pv;
for i = 3:length(h)
    luminosita = 0.8 + 0.2 * (mod(i-3, 2));  % Alterna tra train (0.8) e test (1.0)
    colore = color_ssm * luminosita;
    colore = min(colore, 1);
    h(i).FaceColor = colore;
end
ylim([-30 100])
grid on
xlabel('patient')
ylabel('FIT')
title('FIT')
% Creare la legenda dinamicamente
legenda_base = {'train pv', 'test pv'};
for i = exp_id
    legenda_base{end+1} = ['train ssm_' num2str(i)];
    legenda_base{end+1} = ['test ssm_' num2str(i)];
end
legend(legenda_base);

% ----------------------------------
figure; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO; set(gcf,'Color','w');      % sfondo della figura BIANCO set(gca,'Color','w');      % sfondo dell'asse BIANCO
FIT_all = [ ...
    FIT_train_linear_PV(:), ...
    FIT_test_linear_PV(:), ...
    FIT_train_101(:), ...
    FIT_test_101(:), ...
    FIT_train_SSM, ...
    FIT_test_SSM];
% Creare label dinamicamente
labels_base = {'train pv', 'test pv', 'train 101', 'test 101'};
for i = exp_id
    labels_base{end+1} = ['train ssm_' num2str(i)];
    labels_base{end+1} = ['test ssm_' num2str(i)];
end
boxplot(FIT_all, ...
'Labels', labels_base, ...
'Whisker', 1.5);
ylim([-30 100])
grid on
ylabel('FIT')
title('FIT')
% Colori coerenti
h = findobj(gca,'Tag','Box');
colors_base = {color_pv, color_pv, color_101, color_101};
% Aggiungere train e test per ogni SSM
for i = 3:3+length(exp_id)
    
    luminosita_train =0.8 + 0.2 * (i - 4);
    colore_train = color_ssm * luminosita_train;
    colore_train = min(colore_train, 1);
    colors_base{end+1} = colore_train;
    
    luminosita_test = 0.8 + 0.2 * (i - 4);
    colore_test = color_ssm * luminosita_test;
    colore_test = min(colore_test, 1);
    colors_base{end+1} = colore_test;
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