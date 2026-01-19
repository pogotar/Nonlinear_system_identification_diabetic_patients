close all
set(0, 'DefaultFigureColor', 'w')

% da excel pavia_results

FIT_personalized_linear_model_1day_MPC = [30.59931667
17.23160786
-7.173513518
0.196656186
6.974372866
47.42491792
33.80686733
54.35364511
22.7774801
56.45170532
]';

FIT_personalized_linear_model_2day_MPC = [29.58202158
-32.50074788
3.579809701
-15.82851355
-23.82095675
52.17612065
-13.5154516
55.75019613
2.973768261
59.41815861
]';


FIT_personalized_linear_model_2day_train_batch_1 = [21.75627697
9.846218853
-4.242890544
-11.3287828
-13.39650664
52.23483111
21.5359074
44.22402232
3.367201685
53.6277472
]';

FIT_personalized_linear_model_2day_test = [25.61241932
10.0404122
-2.362629577
-7.458485764
-24.55571974
49.98437939
17.0064642
42.75186167
-0.353463165
51.4619074
]';



figure
% Cambia il colore della terza barra a verde
h = bar([FIT_personalized_linear_model_1day_MPC; FIT_personalized_linear_model_2day_MPC; ...
    FIT_personalized_linear_model_2day_train_batch_1; FIT_personalized_linear_model_2day_test]');
h(1).FaceColor = [0.85 0.8 0.3]; % verde
h(2).FaceColor = [0.75 0.7 0.3]; % verde
h(3).FaceColor = [0.65 0.6 0.3]; % verde
h(4).FaceColor = [0.55 0.5 0.3]; % verde
grid on
xlabel('patient')
ylabel('FIT')
title('error in delta glycemia with just insulin')
legend('MPC data in 1 day','MPC data in 2 day','PID data train part', 'PID data test part')


figure
% Cambia il colore della terza barra a verde
h = bar([FIT_personalized_linear_model_1day_MPC; ...
    FIT_personalized_linear_model_2day_train_batch_1; FIT_personalized_linear_model_2day_test]');
h(1).FaceColor = [0.95 0.85 0.1]; % verde
h(2).FaceColor = [0.85 0.75 0.1]; % verde
h(3).FaceColor = [0.75 0.65 0.1]; % verde
grid on
xlabel('patient')
ylabel('FIT')
title('FIT pavia personalized models')
ylim([-30 100])
legend('MPC data in 1 day','PID data train part', 'PID data test part')


figure

E_m = [ ...
    FIT_personalized_linear_model_1day_MPC(:), ...
    FIT_personalized_linear_model_2day_train_batch_1(:), ...
    FIT_personalized_linear_model_2day_test(:)];

boxplot(E_m, ...
    'Labels', {'MPC data in 1 day','PID data train part', 'PID data test part'}, ...
    'Whisker', 1.5);

grid on
ylabel('FIT')
title('FIT pavia personalized models')

% Colori
h = findobj(gca,'Tag','Box');
colors = {[0.95 0.85 0.1], [0.85 0.75 0.1], [0.75 0.65 0.1]};

for i = 1:length(h)
    patch(get(h(i),'XData'), get(h(i),'YData'), ...
        colors{length(h)-i+1}, ...
        'FaceAlpha',0.6);
end

