close all

error_formula_I = [14.8545
193.3552
18.4718
86.5
106.051
19.5642
81.2424
70.0097
74.2274
78.3447
]';

error_formula_M = [-116.4211
22.2681
56.9607
88.0045
-29.6473
-49.4214
-27.3271
11.9046
-24.8753
45.9408
]';

error_101_I = [31.9405
387.4124
139.5279
202.2413
134.2182
34.4036
91.6276
102.8524
84.4833
164.6721
]';

error_101_M = [-124.9341
-67.0156
-32.1198
-1.7456
-39.2877
-58.7318
-36.5117
-44.6543
-34.3883
-43.5362
]';


error_SSM_I = [14.37509681
27.1619748
-42.89299742
109.6612576
-49.87779421
-21.34447295
-112.283103
-8.71077594
45.3903786
-28.54049773
]';

error_SSM_M = [18.9513454
80.8113488
35.8231474
101.3079722
14.9605427
-14.8311201
84.147091
99.2757188
53.1106403
39.3827304
]';

figure
% Cambia il colore della terza barra a verde
h = bar([abs(error_formula_I); abs(error_101_I); abs(error_SSM_I)]');
h(3).FaceColor = [0 0.7 0]; % verde
grid on
xlabel('patient')
ylabel('error')
title('error in delta glycemia with just insulin')
legend('prior','model 101','SSM')


figure
% Cambia il colore della terza barra a verde
h = bar([abs(error_formula_M); abs(error_101_M); abs(error_SSM_M)]');
h(3).FaceColor = [0 0.7 0]; % verde
grid on
xlabel('patient')
ylabel('error')
title('error in delta glycemia with just meal')
legend('prior','model 101','SSM')


%%

FIT_train = [57.65576935
32.08636475
53.31986618
24.97754669
64.0181427
61.43546677
66.92127228
64.61468506
55.32841492
42.70842743
]';

FIT_test = [56.76078796
43.73545074
51.67752457
29.21724319
55.06151962
65.96082306
66.30969238
66.03881073
67.43339539
63.52604675
]';

figure
h = bar([FIT_train; FIT_test]');

h(1).FaceColor = [0.3 0.8 0]; % verde
h(2).FaceColor = [0.2 0.4 0.2]; % verde
ylim([0 100])
grid on
xlabel('patient')
ylabel('FIT')
title('FIT')
legend('SSM train', 'SSM test')

