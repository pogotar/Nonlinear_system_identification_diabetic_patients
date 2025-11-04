clear all; close all; clc

% daa generate_snc_valencia

%%%%%%%%% tutto in CL !!!!!!

test_days  = 30;  %number of days (training = 17 normale + 4 strani di chiara (21 tot), val = 7, test = 7)
seed = 6; rng(seed,'twister');

config.unannounced_presence  = 1; % 1 se pasti veri diversi da pasti annunciati 

%% meal times

%breakfast
struct_meals.h_break_start  = hours(7); struct_meals.h_break_end  = hours(9.5);
possible_break_h  = [datetime('today') + struct_meals.h_break_start:minutes(15):datetime('today') + struct_meals.h_break_end]; possible_break_h = possible_break_h - datetime('today');

%lunch
struct_meals.h_lunch_start  = hours(11.5); struct_meals.h_lunch_end  = hours(14.5);
possible_lunch_h  = [datetime('today') + struct_meals.h_lunch_start:minutes(15):datetime('today') + struct_meals.h_lunch_end]; possible_lunch_h = possible_lunch_h - datetime('today');

%dinner
struct_meals.h_dinner_start = hours(19); struct_meals.h_dinner_end = hours(22);
possible_dinner_h = [datetime('today') + struct_meals.h_dinner_start:minutes(15):datetime('today') + struct_meals.h_dinner_end]; possible_dinner_h = possible_dinner_h - datetime('today');

%snack 1
struct_meals.h_spunt_1_start = hours(10); struct_meals.h_spunt_1_end = hours(11);
possible_snacks_date_brunch = [datetime('today') + struct_meals.h_spunt_1_start:minutes(15):datetime('today') + struct_meals.h_spunt_1_end]; possible_snacks_date_brunch = possible_snacks_date_brunch - datetime('today');

%snack 2
struct_meals.h_spunt_2_start = hours(15); struct_meals.h_spunt_2_end = hours(18);
possible_snacks_date_lunchinner = [datetime('today') + struct_meals.h_spunt_2_start:minutes(15):datetime('today') + struct_meals.h_spunt_2_end]; possible_snacks_date_lunchinner = possible_snacks_date_lunchinner - datetime('today');
possible_snacks_date_tot = [possible_snacks_date_brunch possible_snacks_date_lunchinner];
 
%% meals

%estrazione time meals
test_h_break = possible_break_h(round(unifrnd( 1, length(possible_break_h),1,test_days )));
test_h_lunch = possible_lunch_h(round(unifrnd( 1, length(possible_lunch_h),1,test_days )));
test_h_dinn = possible_dinner_h(round(unifrnd( 1, length(possible_dinner_h),1,test_days )));

initial_date = datetime('today');

for i = 1:test_days

    time_break(i) = initial_date + days(i-1) + test_h_break(i);
    time_lunch(i) = initial_date + days(i-1) + test_h_lunch(i);
    time_dinn(i)  = initial_date + days(i-1) + test_h_dinn(i);

    tot_g_break(i) = meal_simulation(   time_break(i), struct_meals);
    tot_g_lunch(i) = meal_simulation(   time_lunch(i), struct_meals);
    tot_g_dinn(i) = meal_simulation(  time_dinn(i), struct_meals);
end

tot_datetime = reshape([time_break;time_lunch;time_dinn], [1,3*size(time_break,2)] );
tot_g = reshape([tot_g_break;tot_g_lunch;tot_g_dinn], [1,3*size(tot_g_break,2)] );

%% corrections

% gg_to_correct = 3; % 4-1
% 
% tot_g(17*3+10) = 30;
% 
tot_g_ann = tot_g;
tot_datetime_ann = tot_datetime;
% 
% tot_g_ann(17*3+2) = []; tot_datetime_ann(17*3+2) = [];            % unannounced lunch
% tot_datetime_ann(17*3+5) = tot_datetime_ann(17*3+5) - hours(2); % dinner unannounced 3 hours before 
% tot_datetime_ann(17*3+7) = tot_datetime_ann(17*3+7) + hours(3); % meal announced 3 hours after
% tot_datetime(17*3+10) = []; tot_g(17*3+10) = [];                            % bolus with no meal

%% snacks

% = 0,1,2 how many snacks for each day
how_many_snack_each_day = round(unifrnd( 0,0,1,test_days)); % snacks -> round(unifrnd( 0,2,1,test_days));

%time snacks
tot_snacks_datetime = [];
for i = 1:test_days
    snacks_date=[]; %no snacks
    if how_many_snack_each_day(i) == 1 %1 snack
        snacks_date = possible_snacks_date_tot(round(unifrnd( 1, length(possible_snacks_date_tot) )));
    elseif how_many_snack_each_day(i) == 2 %2 snacks
        snacks_date = [possible_snacks_date_brunch(round(unifrnd( 1, length(possible_snacks_date_brunch) ))),...
                        possible_snacks_date_lunchinner(round(unifrnd( 1, length(possible_snacks_date_lunchinner) )))];
    end
    tot_snacks_datetime = [tot_snacks_datetime initial_date+days(i-1)+snacks_date];
end
% g snacks
tot_snacks_g = [];
for i=1:length(tot_snacks_datetime)
    tot_snacks_g(i) = meal_simulation(   tot_snacks_datetime(i) , struct_meals);
end



%% merge meals + snacks

% veri

combinedDatetime = [tot_snacks_datetime, tot_datetime];
combinedValues = [tot_snacks_g, tot_g];

%Sort the combined datetime vector
[sortedDatetime, sortIdx] = sort(combinedDatetime);

%Reorder the associated values according to the sorted datetime
sortedValues = combinedValues(sortIdx);

tot_g = round(sortedValues)+5;
tot_minutes = sortedDatetime;
tot_minutes = minutes(tot_minutes-initial_date);



% annunciati        

combinedDatetime_ann = [tot_snacks_datetime, tot_datetime_ann];
combinedValues_ann = [tot_snacks_g, tot_g_ann];

%Sort the combined datetime vector
[sortedDatetime_ann, sortIdx_ann] = sort(combinedDatetime_ann);

%Reorder the associated values according to the sorted datetime
sortedValues_ann = combinedValues_ann(sortIdx_ann);

tot_g_ann = round(sortedValues_ann)+5;
tot_minutes_ann = sortedDatetime_ann;
tot_minutes_ann = minutes(tot_minutes_ann-initial_date);


%% scn files

tot_g;
tot_minutes;
Dmeals = 15*ones(size(tot_g));
Dmeals_ann = 15*ones(size(tot_g_ann));

disp('Veri')

disp(['%Tmeals=[' num2str(tot_minutes) ']'])
disp(['%Ameals=[' num2str(tot_g) ']'])
disp(['%Dmeals=[' num2str(Dmeals) ']'])

fprintf('\n')
disp('Annunciati')
disp(['%Tmeals=[' num2str(tot_minutes_ann) ']'])
disp(['%Ameals=[' num2str(tot_g_ann) ']'])
disp(['%Dmeals=[' num2str(Dmeals_ann) ']'])


%% figures

figure
plot(sortedDatetime,tot_g,'bo','MarkerSize',10,'LineWidth',1.5)
grid on
ylim([0 100])
xlim([sortedDatetime(1) sortedDatetime(1)+days(3)])
title('Distribuzione pasti','FontSize',14)
xlabel('Time [hh:mm]','FontSize',12)
ylabel('g CHO','FontSize',12)

figure
plot(hours(mod(tot_minutes/60,24)),tot_g,'bo','MarkerSize',6,'LineWidth',1.5)
hold on
plot(struct_meals.h_break_start*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_break_end*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_lunch_start*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_lunch_end*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_dinner_start*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_dinner_end*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_spunt_1_start*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_spunt_1_end*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_spunt_2_start*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
plot(struct_meals.h_spunt_2_end*ones(90,1),1:90,'Color', [0.5 0.5 0.5],'Linestyle','--')
grid on
xlim([hours(0) hours(24)])
xticks([struct_meals.h_break_start struct_meals.h_break_end...
        struct_meals.h_spunt_1_start struct_meals.h_spunt_1_end...
        struct_meals.h_lunch_start struct_meals.h_lunch_end...
        struct_meals.h_spunt_2_start struct_meals.h_spunt_2_end...
        struct_meals.h_dinner_start struct_meals.h_dinner_end])
title('Distribuzione pasti','FontSize',14)
xlabel('ore [h]','FontSize',12)
ylabel('g CHO','FontSize',12)

%% veri

% 450 780 1200 1845 2175 2426 2775 3315 3645 4065 5040 5176 5460 
% 49 62 87 36 45 15 63 48 60 83 36 25 51
% 
% 
% %% announced
% 
% 450	1200	1845	2175	2595	3495	3645	4065	4710	5040	5460	
% 49		87	36	45	63	48	60	83	49	36	51	

