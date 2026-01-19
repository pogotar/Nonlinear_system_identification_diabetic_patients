% 101 non si può usare perchè anche lui è identificato


clear; close all; clc
%% da modificare
dati_paziente = [1:10];



Ts = 5;     % [s] interpretati come minuti

%%

currend_folder = pwd;
[path_padre, nome_cartella, ~] = fileparts(currend_folder);

% main_folders = ["train", "test"];
main_folders = ["test", "train","prova"];



load(['linearModels_adult_pop20.mat'])



for main_folder = main_folders

    tot_path = path_padre + "\" + main_folder;

    sottocartelle = trovaSottocartelle(tot_path);
    sottocartelle(contains(sottocartelle, '101')) = [];

    for sottocartella = sottocartelle
        if strcmp(sottocartella{1}, 'sc_30days_identification_rwgn') || strcmp(sottocartella{1}, 'sc_1_day_test_IR_meal_80g_13h') || strcmp(sottocartella{1}, 'sc_1_day_test_IR_insulin_60g_15h') || strcmp(sottocartella{1}, 'PAV_EXT_1_100')


            for k = dati_paziente

                folder_data = string(tot_path + "\" + sottocartella{1} );

                disp(folder_data)
                disp(k)

                % load inpulse response model


                model_pers = load("adult#" + num2str(k,'%03i') + "_A.mat", "model");    % carico modello
                model_pers = model_pers.model;
                model_101 = load("adult#" + num2str(101,'%03i') + "_A.mat", "model");

                load(folder_data + "\s#adult#" + num2str(k,'%03i'), "CGM", "injection", "carb_intake", "scenario", 'basal_pattern_original', "Quest", "G")
                load ("C:\Users\pmong\OneDrive - Università di Pavia\EPFL\Nonlinear_system_identification_modified\data\train\sc_2days_identification" + "\s#adult#" + num2str(k,'%03i'), "iAP")

                injection_struct = injection;

                % 1 min    continuo
                injection       = injection_struct.signals.values/Quest.weight; %pmol/min --> pmol/min/Kg
                carb_intake     = carb_intake.signals.values(1:2:end);
                G               = G.signals.values;
                continuous_time = injection_struct.time;

                % equilibri
                eq_CGM_impulse = Quest.Gb;  % rispetto ad usare equil i FIT vengono leggermente meglio
                carb_eq        = 0;

                % basale
                num_giorni = ceil(length(injection)/1440);
                basale_giornaliero = [];
                basal_time_tot = [basal_pattern_original.time 1440];
                for i = 1:length(basal_pattern_original.time)
                    basale_giornaliero = [basale_giornaliero; repmat(basal_pattern_original.values(i), basal_time_tot(i+1)- basal_time_tot(i),1)];
                end
                basale = repmat(basale_giornaliero,num_giorni,1);
                basale = [basale(1); basale];

                % se ricampiono a 5 senz aprima aver tolto il basale ho inizialmente degli zero e quindi devo fare 1:5:end
                % il tutto / Ts

                basale = basale(1:length(injection));

                x_solution = sum(injection(2:24*5)) / sum(basale(2:24*5));
                disp(['x_solution continuo: ' mat2str(x_solution)]);
                % x_solution = [1.7059 1.3325 1.8091 1.3909 1.5337 1.4373 1.4157 1.5321 1.5227 1.2496];
                injection    = injection-x_solution*basale;  % messo per dare in pasto al modello 7 0 se basale e non boli, evidentemente errori ... il discreto dopo è corretto


                % injection    = injection-basale; % !!! togliere 1.3 per come aveva fatto la prof

                %%
                G_sim_tot = lsim(model_pers,[injection,carb_intake],continuous_time); % modello qui è continuo
                G_sim_continuo = G_sim_tot + eq_CGM_impulse;

                %% effetti
                A_i = model_pers.A(1:4,1:4);
                B_i = model_pers.B(1:4,1);
                C_i = model_pers.C(1,1:4);
                D_i = model_pers.D(1,1);

                % matrici dell'inpulse response dei pasti
                A_d = model_pers.A(5:7,5:7);
                B_d = model_pers.B(5:7,2);
                C_d = model_pers.C(1,5:7);
                D_d = model_pers.D(1,2);

                G_sim_i = lsim(ss(A_i, B_i, C_i, D_i),injection,continuous_time);
                G_sim_d = lsim(ss(A_d, B_d, C_d, D_d),carb_intake,continuous_time);

                figure(1)
                plot(continuous_time, G, LineWidth=1.5); hold on; grid on
                plot(continuous_time, G_sim_tot+eq_CGM_impulse)
                plot(continuous_time, G_sim_i+eq_CGM_impulse)
                plot(continuous_time, G_sim_d+eq_CGM_impulse)
                plot(continuous_time, G_sim_i + G_sim_d+eq_CGM_impulse, LineWidth=1.5)
                legend({'G vera','G tot', 'G i', 'G d','G i + G d'}); title(['patient: ' num2str(k)]);

                % ylim([30 400])
                % xlim([1 3*24*60])

                hold off

                % folder_to_save = "./../../" + type_of_source + "/saved_G_i_G_sum_G_d/" + code_version;
                %
                % if ~exist(folder_to_save, 'dir')
                %     % Create the folder if it does not exist
                %     mkdir(folder_to_save);
                %     disp(['Folder "', folder_to_save, '" has been created.']);
                % end


                % save( folder_to_save + "/s#adult#" + num2str(k,'%03i'), "G_sim_tot", "G_sim_i","G_sim_d","eq_CGM_impulse")



                %% discreto

                m_bar=equil(k).Ueq(1); % u eq
                i_bar=equil(k).Ueq(2); % u eq
                G_bar= equil(k).Yeq;   % y eq
                x_bar= equil(k).Xeq;   % x eq


                load(folder_data + "\s#adult#" + num2str(k,'%03i'), "CGM", "injection", "carb_intake", "scenario", 'basal_pattern_original', "Quest", "G")

                model = c2d(model_pers,Ts);


                % 1 min    continuo
                carb_intake_2            = carb_intake.signals.values(1:2*Ts:end);
                carb_intake     = carb_intake.signals.values(1:2:end);
                G               = G.signals.values;
                continuous_time = injection_struct.time;

                % equilibri
                eq_CGM_impulse = Quest.Gb;  % rispetto ad usare equil i FIT vengono leggermente meglio
                carb_eq        = 0;


                % resample
                carb_intake_discreto     = carb_intake(2:Ts:end); % gi' estratto dalla struttura
                G_discreto               = G(2:Ts:end); % % gi' estratto dalla struttura
                time_discreto            = continuous_time(2:Ts:end);


                %% injection
                % è corretto questo ma non era coerente con quello fatto in
                % python

                % injection_struct = injection;
                %
                % % 1 min    continuo
                % injection       = injection_struct.signals.values/Quest.weight; %pmol/min --> pmol/min/Kg
                %
                % % basale
                % num_giorni = ceil(length(injection)/1440);
                % basale_giornaliero = [];
                % basal_time_tot = [basal_pattern_original.time 1440];
                % for i = 1:length(basal_pattern_original.time)
                %     basale_giornaliero = [basale_giornaliero; repmat(basal_pattern_original.values(i), basal_time_tot(i+1)- basal_time_tot(i),1)];
                % end
                % basale = repmat(basale_giornaliero,num_giorni,1);
                % basale = [basale(1); basale];
                % basale = basale(1:length(injection));
                %
                %
                %
                % % x_solution = sum(injection(2:6)) / sum(basale(2:6));
                % % injection    = injection-x_solution*basale;
                % injection    = injection-basale;
                %
                % sum(injection(2:6))
                %
                % injection_discreto = [];
                % for i=2:Ts:length(injection)
                %     injection_discreto = [injection_discreto; sum(injection(i:i+Ts-1))];
                % end
                %
                %
                % injection_discreto_true = injection_discreto/Ts;

                %% injection
                % è approssimato (nel senso che approssima quello che entra nei modelli mentre è perfetto per tutta la pipeline python)

                % basal_pattern_original  per openloop o pid
                % basalBolusMem se mpc
                % 1 min    continuo
                injection       = injection_struct.signals.values(2:5:end)/6000; % U/5min
                basal_new = [];
                basal_tot = [];
                for i = 1:length(injection)
                    % Time of Day calculation
                    ToD = mod(i * 5 - 1, 1440);

                    % Find indices where basal_time <= ToD
                    indices = find(basal_pattern_original.time <= ToD);

                    if ~isempty(indices)
                        currentBasal = basal_pattern_original.values(indices(end));  % ultimo valore valido
                    else
                        currentBasal = basal_pattern_original.values(end);  % se nessun valore valido, prendi l'ultimo
                    end

                    basal = currentBasal;
                    basal_tot = [basal_tot basal];
                end

                basal_tot = basal_tot'/ 60 * 5; % U

                x_solution = sum(injection(2:24)) / sum(basal_tot(2:24));
                disp(['x_solution discreto: ' mat2str(x_solution)])
                x_solution = 1;

                injection = injection*(1/Ts)*6000/Quest.weight; % pmol/5min/kg

                basal_tot = basal_tot*(1/Ts)*6000/Quest.weight; % pmol/5min/kg

                injection_discreto = injection - basal_tot;



                % figure; plot(injection_discreto_true); hold on ; plot(injection_discreto_2); legend('1','2')


                %%

                %
                G_sim = lsim(model,[injection_discreto,carb_intake_discreto],time_discreto); % modello qui [ continuo
                G_sim_discreto= G_sim + eq_CGM_impulse;


                %%
                figure(10)
                plot(time_discreto  , G_discreto);             hold on; grid on
                plot(continuous_time, G_sim_continuo, LineWidth=5)
                plot(time_discreto  , G_sim_discreto,LineWidth=2)
                % plot(time_discreto  , G_sim_2,LineWidth=2)

                legend({'G', 'G simulato continuo', 'G simulato discreto'});

                title(['patient: ' num2str(k)]);

                hold off

                %%

                numerator = norm(G_sim_discreto - G_discreto, 2);
                denominator = norm(G_discreto - mean(G_discreto), 2);
                FIT = 100 * (1 - numerator / denominator);

                numerator = norm(G_sim_discreto(1:1440/5) - G_discreto(1:1440/5), 2);
                denominator = norm(G_discreto(1:1440/5) - mean(G_discreto(1:1440/5)), 2);
                FIT_1day = 100 * (1 - numerator / denominator);

                % Calculate MSE (Mean Squared Error)
                MSE = mean((G_sim_discreto - G_discreto).^2);

                carb_plot = zeros(size(carb_intake_discreto));

                for i = 1:15:length(carb_intake_discreto)-15
                    carb_plot(i) = sum(carb_intake_discreto(i:i+15));
                end

                carb_plot = carb_plot/1000*Ts; % g


                plot_glucose_insulin('meal', carb_plot, 'glucose', G_discreto, 'predicted_glucose', G_sim_discreto, ...
                    'insulin', injection_discreto, 'title', ['patient:' num2str(k) '  |  FIT: ' num2str(FIT)  '  |  MSE: ' num2str(MSE) ]);

                disp(['patient: ' num2str(k)   '  |  FIT: ' num2str(FIT)  ' |  FIT_1day: ' num2str(FIT_1day)  '  |  MSE: ' num2str(MSE)  '  |  dataset: ' sottocartella{1}])

                [delta_I, idx_i] = max(injection_discreto);
                delta_I = delta_I * Quest.weight / 6000 * 5;
                disp(['delat I:  ' num2str(delta_I) ' [U] '])
                CF = iAP.RCM_param.CFpatientForModel;


                [delta_M, idx_m] = max(carb_plot);
                disp(['delat M:  ' num2str(delta_M)])



                CRtv = load("CRtv.mat");

                %%


                if strcmp(sottocartella{1}, 'PAV_EXT_1_100')
                    % PAV_EXT_1_100 -> with_MPC in excel
                    row = 1 + 0 * 10 + k;

                    modifiche = struct();
                    modifiche.FIT_personalized_linear_model_1day = {row, FIT_1day};
                    modify_xlsx_row_and_column('pavia_results.xlsx', modifiche);

                    modifiche = struct();
                    modifiche.FIT_personalized_linear_model_2day = {row, FIT};
                    modify_xlsx_row_and_column('pavia_results.xlsx', modifiche);

                    disp('ciao')
                end



                if strcmp(sottocartella{1}, 'sc_30days_identification_rwgn')
                    % sc_30days_identification_rwgn -> train_batch_1 and
                    % test_similar_to_train in excel

                    start_valid = 5*12;
                    num_days = 30;
                    train_size = num_days*0.8*1440/5;
                    val_size = num_days*0.1*1440/5;
                    test_size = num_days*0.1*1440/5;
                    batch_size = 12;

                    tot_train = (start_valid+train_size)/batch_size;
                    tot_valid_s = start_valid+train_size;
                    tot_valid_f = start_valid+train_size+val_size;
                    tot_test = start_valid+train_size+val_size;


                    G_sim_discreto_train = lsim(model_pers,[injection_discreto(1:tot_train),carb_intake_discreto(1:tot_train)],time_discreto(1:tot_train)) + eq_CGM_impulse;
                    G_sim_discreto_valid = lsim(model_pers,[injection_discreto(tot_valid_s:tot_valid_f),carb_intake_discreto(tot_valid_s:tot_valid_f)],time_discreto(tot_valid_s:tot_valid_f)) + eq_CGM_impulse;
                    G_sim_discreto_test = lsim(model_pers,[injection_discreto(tot_test:end),carb_intake_discreto(tot_test:end)],time_discreto(tot_test:end)) + eq_CGM_impulse;

                    disp(length(G_sim_discreto_train))
                    disp(length(G_sim_discreto_valid))
                    disp(length(G_sim_discreto_test))


                    G_train = G_discreto(1:tot_train);
                    G_valid = G_discreto(tot_valid_s:tot_valid_f);
                    G_test = G_discreto(tot_test:end);

                    numerator = norm(G_sim_discreto_train - G_train, 2);
                    denominator = norm(G_train - mean(G_train), 2);
                    FIT_train = 100 * (1 - numerator / denominator);

                    numerator = norm(G_sim_discreto_test - G_test, 2);
                    denominator = norm(G_test - mean(G_test), 2);
                    FIT_test = 100 * (1 - numerator / denominator);

                    disp(['FIT_train:' num2str(FIT_train)])
                    disp(['FIT_test:' num2str(FIT_test)])

                    row_train = 1 + 1 * 10 + k;
                    row_test = 1 + 2 * 10 + k;

                    modifiche = struct();
                    modifiche.FIT_personalized_linear_model_2day = {row_train, FIT_train};
                    modify_xlsx_row_and_column('pavia_results.xlsx', modifiche);

                    modifiche = struct();
                    modifiche.FIT_personalized_linear_model_2day = {row_test, FIT_test};
                    modify_xlsx_row_and_column('pavia_results.xlsx', modifiche);

                    disp('ciao')

                end


                if strcmp(sottocartella{1}, 'sc_1_day_test_IR_meal_80g_13h') || strcmp(sottocartella{1}, 'sc_1_day_test_IR_insulin_60g_15h')

                    if delta_M == 0
                        % ins
                        row = 1 + 3 * 10 + k;
                        [G_min, idx_min] = min(G_discreto(idx_i:idx_i+6*60/5)); % guarda nelle successive 6 ore
                        delta_G_empirico = G_discreto(idx_i) - G_min;
                        delta_G_modello = G_sim_discreto(idx_i) - min(G_sim_discreto);
                        disp(['delta G empirico:  ' num2str(delta_G_empirico)])
                        disp(['delta G formula:  ' num2str(CF * delta_I)])
                        disp(['delta G modello:  ' num2str(delta_G_modello)])

                        disp(['delta t modello:' num2str((idx_min-idx_i)*5)])

                        modifiche = struct();
                        modifiche.delta_G_linear_personalized = {row, delta_G_modello};
                        modify_xlsx_row_and_column('pavia_results.xlsx', modifiche);

                    else
                        % meal
                        row = 1 + 4 * 10 + k;
                        [G_max, idx_max] = max(G_discreto(idx_m:idx_m+6*60/5)); % guarda nelle successive 6 ore
                        delta_G_empirico = G_max - G_discreto(idx_m) ;
                        delta_G_modello = max(G_sim_discreto) - G_sim_discreto(idx_m);
                        disp(['delta G empirico:  ' num2str(delta_G_empirico)])

                        CR_min = min(CRtv.CRtv(k).values);
                        CR_max = max(CRtv.CRtv(k).values);
                        CR = iAP.RCM_param.CRpatientForModel;

                        disp(['delta G formula :  ' num2str(CF/CR * delta_M)])

                        % disp(['delat G formula max:  ' num2str(CF/CR_min * delta_M)])
                        % disp(['delat G formula min:  ' num2str(CF/CR_max * delta_M)])

                        disp(['delta G modello:  ' num2str(delta_G_modello)])

                        disp(['delta t modello:' num2str((idx_max-idx_m)*5)])

                        modifiche = struct();
                        modifiche.delta_G_linear_personalized = {row, delta_G_modello};
                        modify_xlsx_row_and_column('pavia_results.xlsx', modifiche);

                    end

                    disp('ciao')

                end


                disp('ciao')


            end
        end

    end

end

%% Funzione: modify_xlsx_row_and_column
function modify_xlsx_row_and_column(file_path, modifiche)
% Modifica valori cercando la colonna per NOME (dalla prima riga).
% Crea automaticamente le colonne mancanti nella posizione disponibile.
%
% Args:
%     file_path: percorso del file .xlsx (es: 'dati.xlsx')
%     modifiche: struct dove:
%                - field: nome della colonna (da prima riga)
%                - value: struct con {numero_riga: nuovo_valore}
%
% Esempio:
%     modifiche = struct();
%     modifiche.Nome = struct('2', 'Marco', '5', 'Luca');
%     modifiche.Eta = struct('2', 30, '5', 25);
%     modify_xlsx_row_and_column('dati.xlsx', modifiche);

% Carica il file Excel
[num, txt, raw] = xlsread(file_path);

% Leggi la prima riga per trovare i nomi delle colonne
intestazione = containers.Map;
num_cols = size(raw, 2);

for col_idx = 1:num_cols
    cell_value = raw{1, col_idx};
    if ~isempty(cell_value) && ischar(cell_value)
        intestazione(cell_value) = col_idx;
    elseif ~isempty(cell_value) && isstring(cell_value)
        intestazione(char(cell_value)) = col_idx;
    end
end

% Trova la prossima colonna disponibile
if isempty(intestazione)
    prossima_col_libera = 1;
else
    prossima_col_libera = max(cell2mat(intestazione.values())) + 1;
end

% Leggi i nomi delle colonne da creare
nomi_colonne = fieldnames(modifiche);

% Crea le colonne mancanti nell'ordine in cui le passi
for i = 1:length(nomi_colonne)
    nome_colonna = nomi_colonne{i};

    if ~isKey(intestazione, nome_colonna)
        raw{1, prossima_col_libera} = nome_colonna;
        intestazione(nome_colonna) = prossima_col_libera;
        fprintf('✓ Colonna ''%s'' creata nella posizione %d\n', nome_colonna, prossima_col_libera);
        prossima_col_libera = prossima_col_libera + 1;
    end
end

% Modifica le celle
for i = 1:length(nomi_colonne)
    nome_colonna = nomi_colonne{i};
    dati = modifiche.(nome_colonna);
    col_idx = intestazione(nome_colonna);

    % Se è una cell array {riga, valore}
    if iscell(dati) && length(dati) == 2
        riga = dati{1};
        valore = dati{2};

        % Espandi raw se necessario
        if riga > size(raw, 1)
            raw(riga, col_idx) = {valore};
        else
            raw{riga, col_idx} = valore;
        end
        fprintf('Modificato %s (riga %d): %s\n', nome_colonna, riga, num2str(valore));
    end
end

% Salva il file
xlswrite(file_path, raw);
fprintf('\n✓ File %s aggiornato con successo\n', file_path);
end

function plot_glucose_insulin(varargin)
% Plot Glucose/Meal (top) and Insulin (bottom, if present) with dual y-axes.
%
% Syntax:
%   plot_glucose_insulin('insulin', insulin_array, 'meal', meal_array, ...)
%
% Parameters:
%   insulin           - insulin array (optional, 1D)
%   meal              - meal array (optional, 1D)
%   glucose           - actual glucose array (optional, 1D)
%   predicted_glucose - predicted glucose array (optional, 1D)
%   title             - plot title (default: 'Glucose and Insulin vs Time')

% Parse input arguments
p = inputParser;
addParameter(p, 'insulin', [], @isvector);
addParameter(p, 'meal', [], @isvector);
addParameter(p, 'glucose', [], @isvector);
addParameter(p, 'predicted_glucose', [], @isvector);
addParameter(p, 'title', 'Glucose and Insulin vs Time', @ischar);
parse(p, varargin{:});

insulin = p.Results.insulin;
meal = p.Results.meal;
glucose = p.Results.glucose;
predicted_glucose = p.Results.predicted_glucose;
plot_title = p.Results.title;

% Create time array based on length of available data
if ~isempty(meal)
    n_points = length(meal);
elseif ~isempty(glucose)
    n_points = length(glucose);
elseif ~isempty(predicted_glucose)
    n_points = length(predicted_glucose);
else
    n_points = length(insulin);
end
time = 1:n_points;

% Determine number of subplots
if isempty(insulin)
    n_plots = 1;
else
    n_plots = 2;
end

% Create figure with appropriate size and layout
if n_plots == 2
    fig = figure('Position', [100, 100, 1000, 525]);
    ax1 = subplot(2, 1, 1);
    ax3 = subplot(2, 1, 2);
    % Set height ratios (top plot larger than bottom)
    set(ax1, 'Position', [0.1 0.55 0.8 0.4]);
    set(ax3, 'Position', [0.1 0.1 0.8 0.35]);
else
    fig = figure('Position', [100, 100, 1000, 350]);
    ax1 = subplot(1, 1, 1);
end

% ===== FIRST PLOT: Meal/Glucose =====

% Remove x-axis labels from first plot if insulin plot exists
if n_plots == 2
    set(ax1, 'XTickLabel', []);
else
    xlabel(ax1, 'Time step', 'FontSize', 11);
end

% Plot Meal on left y-axis
if ~isempty(meal)
    hold(ax1, 'on');
    % Find indices where meal is non-zero
    nonzero_idx = meal ~= 0;
    % Scatter plot only for non-zero values
    scatter(ax1, time(nonzero_idx), meal(nonzero_idx), 50, [60 179 113]/255, ...
        'filled', 'DisplayName', 'Meal', 'MarkerEdgeColor', 'none');
    ylabel(ax1, 'Meal (g)', 'Color', [60 179 113]/255, 'FontSize', 11);
    set(ax1, 'YColor', [60 179 113]/255);
    ax1.YAxis.Color = [60 179 113]/255;
end

% Create second y-axis for glucose (right side)
ax2 = axes('Position', get(ax1, 'Position'), ...
    'XAxisLocation', 'bottom', ...
    'YAxisLocation', 'right', ...
    'Color', 'none', ...
    'XTick', [], ...
    'Parent', fig);

hold(ax2, 'on');
linkaxes([ax1, ax2], 'x');

ylabel(ax2, 'Glucose (mg/dL)', 'Color', [0 0 1], 'FontSize', 11);
set(ax2, 'YColor', [0 0 1]);
ax2.YAxis.Color = [0 0 1];

% Plot predicted glucose
if ~isempty(predicted_glucose)
    plot(ax2, time, predicted_glucose, 'Color', [0 0 0.5], 'LineWidth', 2, ...
        'DisplayName', 'Predicted Glucose', 'LineStyle', '-');
end

% Plot actual glucose
if ~isempty(glucose)
    plot(ax2, time, glucose, 'Color', [100 149 237]/255, 'LineWidth', 2, ...
        'DisplayName', 'Glucose', 'LineStyle', '-');
end

% Grid and title
grid(ax1, 'on');
ax1.GridAlpha = 0.3;
title(ax1, plot_title, 'FontSize', 14, 'FontWeight', 'bold');

% Set y-limits for meal (left axis)
if ~isempty(meal)
    set(ax1, 'YLim', [0, 100]);
end

% Set y-limits for glucose (right axis)
if ~isempty(glucose) || ~isempty(predicted_glucose)
    min_val = 30;
    max_val = 300;

    if ~isempty(glucose) && ~isempty(predicted_glucose)
        data_min = min(min(glucose), min(predicted_glucose));
        data_max = max(max(glucose), max(predicted_glucose));
    elseif ~isempty(glucose)
        data_min = min(glucose);
        data_max = max(glucose);
    else
        data_min = min(predicted_glucose);
        data_max = max(predicted_glucose);
    end

    % Adjust limits if data exceeds bounds
    if data_min < min_val
        min_val = data_min * 0.95;
    end

    % Choose upper limit based on max data value
    if data_max <= 100
        max_val = 100;
    elseif data_max <= 200
        max_val = 200;
    else
        max_val = 300;
    end

    % Add 5% margin if data exceeds chosen limit
    if data_max > max_val
        max_val = data_max * 1.05;
    end

    set(ax2, 'YLim', [min_val, max_val]);
end

% Combine legends from both axes
legend(ax2, 'Location', 'northeast', 'FontSize', 10);

% ===== SECOND PLOT: Insulin (if present) =====
if n_plots == 2
    axes(ax3);
    hold(ax3, 'on');
    plot(ax3, time, insulin, 'Color', [1 0 0], 'LineWidth', 2, ...
        'DisplayName', 'Insulin', 'LineStyle', '-');
    xlabel(ax3, 'Time step', 'FontSize', 11);
    ylabel(ax3, 'Insulin (u1)', 'Color', [1 0 0], 'FontSize', 11);
    set(ax3, 'YColor', [1 0 0]);
    ax3.YAxis.Color = [1 0 0];
    set(ax3, 'YLim', [-1, 100]);
    grid(ax3, 'on');
    ax3.GridAlpha = 0.3;
    legend(ax3, 'Location', 'northeast', 'FontSize', 10);

    % Link x-axis with top plot
    linkaxes([ax1, ax3], 'x');
end
end


