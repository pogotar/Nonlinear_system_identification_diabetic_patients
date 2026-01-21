* preprocessing dati esattamente come nella simulazione dei modelli lineari?

fare  ottimizzazioni di meal e insulina separate?

mettere la loss di non avere spike improvvisi?

(magari rispetto alla somma tutte le predizioni devono essere > 0 o di un agressive factor molto blando?)

oppure penalizzo derivate alte

magari considerare la somma dell’insulina rispetto al primo quartile

calcolare delta di glicemia rispetto G_bar o rispetto all’istante attuale?


---

```python
# train_batch_1   test_similar_to_train   insulin_impulse  meal_impulse

 #scenario    patients    delta_G_empiric    delta_G_101_linearized    delta_G_formula    delta_G_SSM  delta_t_max    FIT_101_linearized    FIT_SSM
```

sc_30days_identification_rwgn   - >  train_batch_1

test_similar_to_train

```python
sc_1_day_test_IR_meal_80g_13h
```

```python
sc_1_day_test_IR_insulin_60g_15h
```


---

tabella results parte di pavia PV generata con

C:\\Users\\pmong\\OneDrive - Università di Pavia\\EPFL\\Nonlinear_system_identification_modified\\data\\linear_models\\ <mark>simulate_model_v2_final</mark>

e salvati in

C:\\Users\\pmong\\OneDrive - Università di Pavia\\EPFL\\Nonlinear_system_identification_modified\\data\\linear_models\\pavia_results.xlsx

parte di empirco e modello fatto copiando e incollando a mano da file <mark>simulate_model_v2_final</mark>

(in simulate_model_v2_final  viene fatto girare il modello 7 originale fatto in acc)

(in <mark>simulate_model_101</mark>  e viene fatto girare il modello 101)

simulate_model_101 -> è corretto nel processamento dati in discreto

simulate_model_v2_final -> è corretto nel processamento dati in discreto


---

| code experiment | description |
|----|----|
| exp_3 | baseline, 10 patients initial implementation, \n (look at exp 5, there  were some adjustments to the code) |
| ==exp_4== | - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 1 |
| ==exp_5== | - present rwgn  → (utils_SSM  use_noise = True) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 1 |
| exp_6 | - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min (insulina, pat i) → 0.4 \n - model_folder_101 = ‘6‘ \n prior loss = 1 |
| exp_7 | - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min (insulina, pat i) → 0.8 \n - rmax → 0.95 \n - model_folder_101 = ‘6‘ \n prior loss = 1 |
| exp_8 | copiata da 4 ma prior loss pesata 0.1 \n - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 0.1 |
| exp_9 | copiata da 4 ma prior loss pesata 0.01 \n - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 0.01 |
| exp_10 | copiata da 9 \n tranne che moltiplicato per 1 e    error > 0, \n 1000 \\\* torch.abs(error),  # errore positivo: 1000 |
| 1.0 \\\* torch.abs(error)   # errore negativo: pesato1 |    |
| ) \| |    |

pensare se pinball

pensare se iob e quanto mettere di tempo assestamento 4 o 5 ore

pensare se mettere tempo assestamento 4 ore e tenere sum e basta



se non funziona nulla taglio forza CR (magari mascherandola con pinball loss)

provare con scenario 40 g  -→ probabilmente il modello non lineare satura → provare con 200 grammi a vedere cosa fa la G del modello non lineare

sc_1_day_test_IR_insulin_30g_13h

sc_1_day_test_IR_insulin_60g_15h

10 + sono con il test insulina meno forte




---

34 → come 4 ma strategy 1 and 2 + test meno forte su insulina (30 g al posto di 60)



44  come 34 ma con prior su derivata