function [CHO] = meal_simulation(T, struct_meals)

T = minutes(T-dateshift(T, 'start', 'day'));

delta_sigma=2.5; % aggiunto da paolo

%breakfast
if T>= minutes(struct_meals.h_break_start) && T <= minutes(struct_meals.h_break_end)

    mu = 45; sigma = 10*delta_sigma;
    CHO = mu + sqrt(sigma)*randn();

    if CHO < 20
        CHO = 20;
    end

%snack 1
elseif T>= minutes(struct_meals.h_spunt_1_start) && T <= minutes(struct_meals.h_spunt_1_end)

    mu = 20;  sigma = 5;
    CHO = mu + sqrt(sigma)*randn();

%lunch
elseif T>= minutes(struct_meals.h_lunch_start) && T <= minutes(struct_meals.h_lunch_end)

    mu = 60; sigma = 10*delta_sigma;
    CHO = mu + sqrt(sigma)*randn();

    if CHO < 20
        CHO = 20;
    end
 
%snack 2
elseif T>=minutes(struct_meals.h_spunt_2_start) && T <= minutes(struct_meals.h_spunt_2_end) 

    mu = 20; sigma = 5;
    CHO = mu + sqrt(sigma)*randn();

%dinner
elseif  T>=minutes(struct_meals.h_dinner_start) && T <= minutes(struct_meals.h_dinner_end)

    mu = 75; sigma = 30;
    CHO = mu + sqrt(sigma)*randn();

    if CHO < 20
        CHO = 20;
    end

end