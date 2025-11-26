clear; close all; clc
%% da modificare
dati_paziente = [1];



Ts = 5;     % [s] interpretati come minuti

%%
% load inpulse response model
load ("PAV#007#iAP.mat", "model")    % carico modello

for k = dati_paziente

    load("s#adult#" + num2str(k,'%03i'), "CGM", "injection", "carb_intake", "scenario", 'basal_pattern_original', "Quest", "G")

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

    % se ricampiono a 5 senz aprima aver tolto il basale ho inizialmente degli zero e quindi devo vare 1:5:end 
    % il tutto / Ts

    basale = basale(1:length(injection));



    injection    = injection-basale;
    %%
    G_sim_tot = lsim(model,[injection,carb_intake],continuous_time); % modello qui è continuo

     %% effetti
    A_i = model.A(1:4,1:4);
    B_i = model.B(1:4,1);
    C_i = model.C(1,1:4);
    D_i = model.D(1,1);

    % matrici dell'inpulse response dei pasti
    A_d = model.A(5:7,5:7);
    B_d = model.B(5:7,2);
    C_d = model.C(1,5:7);
    D_d = model.D(1,2);

    G_sim_i = lsim(ss(A_i, B_i, C_i, D_i),injection,continuous_time);
    G_sim_d = lsim(ss(A_d, B_d, C_d, D_d),carb_intake,continuous_time);

    figure(1)
    plot(continuous_time, G, LineWidth=1.5); hold on; grid on
    plot(continuous_time, G_sim_tot+eq_CGM_impulse)
    plot(continuous_time, G_sim_i+eq_CGM_impulse)
    plot(continuous_time, G_sim_d+eq_CGM_impulse)
    plot(continuous_time, G_sim_i + G_sim_d+eq_CGM_impulse, LineWidth=1.5)
    legend({'G vera','G tot', 'G i', 'G d','G i + G d'})

    ylim([30 400])
    xlim([1 3*24*60])

    hold off
    
    % folder_to_save = "./../../" + type_of_source + "/saved_G_i_G_sum_G_d/" + code_version;
    % 
    % if ~exist(folder_to_save, 'dir')
    %     % Create the folder if it does not exist
    %     mkdir(folder_to_save);
    %     disp(['Folder "', folder_to_save, '" has been created.']);
    % end


    % save( folder_to_save + "/s#adult#" + num2str(k,'%03i'), "G_sim_tot", "G_sim_i","G_sim_d","eq_CGM_impulse")



    % %% discreto
    % 
    % load ("./things_for_superposition_effects/PAV#007#iAP.mat", "model")    % carico modello
    % load("./temp/s#adult#" + num2str(dati_paziente,'%03i'), "CGM", "injection", "carb_intake", "scenario", 'basal_pattern_original', "Quest", "G")
    % 
    % model = c2d(model,Ts);
    % 
    % 
    % % 1 min    continuo
    % carb_intake     = carb_intake.signals.values(1:2:end);
    % G               = G.signals.values;
    % continuous_time = injection_struct.time;
    % 
    % % equilibri
    % eq_CGM_impulse = Quest.Gb;  % rispetto ad usare equil i FIT vengono leggermente meglio
    % carb_eq        = 0;
    % 
    % 
    % % resample
    % carb_intake_discreto     = carb_intake(2:Ts:end); % gi' estratto dalla struttura
    % G_discreto               = G(2:Ts:end); % % gi' estratto dalla struttura
    % time_discreto            = continuous_time(2:Ts:end);
    % 
    % 
    % %% injection
    % injection_struct = injection;
    % 
    % % 1 min    continuo
    % injection       = injection_struct.signals.values/Quest.weight; %pmol/min --> pmol/min/Kg
    % 
    % % basale
    % num_giorni = floor(length(injection)/1440);
    % basale_giornaliero = [];
    % basal_time_tot = [basal_pattern_original.time 1440];
    % for i = 1:length(basal_pattern_original.time)
    %     basale_giornaliero = [basale_giornaliero; repmat(basal_pattern_original.values(i), basal_time_tot(i+1)- basal_time_tot(i),1)];
    % end
    % basale = repmat(basale_giornaliero,num_giorni,1);
    % basale = [basale(1); basale];
    % 
    % 
    % injection    = injection-basale;
    % 
    % injection_discreto = [];
    % for i=2:Ts:length(injection)
    %     injection_discreto = [injection_discreto; sum(injection(i:i+Ts-1))];
    % end
    % 
    % 
    % injection_discreto = injection_discreto/Ts;
    % 
    % %%
    % 
    % %
    % G_sim = lsim(model,[injection_discreto,carb_intake_discreto],time_discreto); % modello qui [ continuo
    % G_sim_discreto= G_sim + eq_CGM_impulse;
    % 
    % 
    % %%
    % figure(10)
    % plot(time_discreto  , G_discreto);             hold on; grid on
    % plot(continuous_time, G_sim_continuo)
    % plot(time_discreto  , G_sim_discreto)
    % 
    % legend({'G', 'G simulato continuo', 'G simulato discreto'});

end


