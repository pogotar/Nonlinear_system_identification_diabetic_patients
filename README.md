tensorboard --logdir=.\\runs\\exp_101 --port=6006

./ngrok http 6006



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
| exp_4 | - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 1 |
| exp_5 | - present rwgn  → (utils_SSM  use_noise = True) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 1 |
| exp_6 | - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min (insulina, pat i) → 0.4 \n - model_folder_101 = ‘6‘ \n prior loss = 1 |
| exp_7 | - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min (insulina, pat i) → 0.8 \n - rmax → 0.95 \n - model_folder_101 = ‘6‘ \n prior loss = 1 |
| exp_8 | copiata da 4 ma prior loss pesata 0.1 \n - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 0.1 |
| exp_9 | copiata da 4 ma prior loss pesata 0.01 \n - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - only strategy 1 \n - r_min  (insulina, pat i) → 0 \n prior loss = 0.01 |
| exp_10 | copiata da 9 \n tranne che moltiplicato per 1 e    error > 0, \n 1000 \\\* torch.abs(error),  # errore positivo: 1000 |
| 1.0 \\\* torch.abs(error)   # errore negativo: pesato1 |    |
| ==exp_34== | come 4 ma strategy 1 e anche 2 ora  + test meno forte su insulina (30 g al posto di 60) \n - no rwgn  → (utils_SSM  use_noise = False) \n - prior just on insulin and same fixed regularization for all patients \n - strategy 1 and==2== \n - r_min  (insulina, pat i) → 0 \n - prior loss insulina = 1 \n  \n !!!! da rifare, per sbaglio 200 epoche |
| exp_44 | come 34 ma prior anche su derivata \n - no rwgn  → (utils_SSM  use_noise = False) \n - strategy 1 \n - r_min  (insulina, pat i) → 0 \n - prior loss insulina = 1 \n - prior derivata (solo insulina) → 1e3 |
| exp_45 | come 34 ma prior anche su derivata \n - no rwgn  → (utils_SSM  use_noise = False) \n - strategy 1 \n - r_min  (insulina, pat i) → 0 \n - prior loss insulina = 1 \n - prior derivata (solo insulina) → 1e2 |
| exp_46 | come 34 ma prior anche su derivata \n - no rwgn  → (utils_SSM  use_noise = False) \n - strategy 1 and 2 \n - r_min  (insulina, pat i) → 0 \n - prior loss insulina = 1 \n - prior derivata (solo insulina) → 1e1 |
| exp_54 | come 34 ma MA alla fine \n - no rwgn  → (utils_SSM  use_noise = False) \n - strategy 1 \n - r_min  (insulina, pat i) → 0 \n !!! dual SSM da mettere a posto se strategy 2 |
| exp_64 | come 34 ma prior insulina 4h |
| exp_74 | come 34 ma prior insulina 4h e preprocessamento iob in loss |
| exp_84 | come 34 \n lowpass filter |
| exp_65 | come 34 ma \n - prior insulina 4h (come in 64) \n - lowpass (come in 84) |
| exp_75 | come 34 ma \n - prior insulina 4h  e preprocessamento iob in loss (come in 74) \n - lowpass (come in 84) |

fare anche test con picco (guardo la più grande differenza consecutiva tra -1 0, 0 1, 1 2, 2 3)  e salvarlo in results_v4

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


---

leonardo dice di

l2n → tv

l2n → tv e gamma


---

75

```

config = {
    
    'code_identifier' : code_identifier,
    'exp_identifier' : exp_identifier,
    'timestamp': datetime.now().isoformat(),
    
    # ===== EPOCHS =====
    'epochs_101': 2000,
    'epochs_s1' : 2000,
    'epochs_s2' : 1000,
    
    # ===== HYPERPARAMETERS =====
    'learning_rate' : 1e-3,
    'use_noise' : False,
    
    # ===== monotonic gain loss =====
    'use_monotonic_gain_loss' : True,
    'cumulative_window' : 12*4,             # 12*2.5    12*4    !!!!!
    'horizon' : 12*0.5,
    'type_preprocess_insulino' : 'iob',      # 'sum'   'iob'    !!!!!!
    
    # ===== monotonic gain loss =====
    'use_low_pass_I' : True,   # per ora implementato solo in strategy 1
    
}
```


---

84

```
config = {
    
    'code_identifier' : code_identifier,
    'exp_identifier' : exp_identifier,
    'timestamp': datetime.now().isoformat(),
    
    # ===== EPOCHS =====
    'epochs_101': 2000,
    'epochs_s1' : 2000,
    'epochs_s2' : 1000,
    
    # ===== HYPERPARAMETERS =====
    'learning_rate' : 1e-3,
    'use_noise' : False,
    
    # ===== monotonic gain loss =====
    'use_monotonic_gain_loss' : True,
    'cumulative_window' : 12*2.5,             # 12*2.5    12*4     !!!!!
    'horizon' : 12*0.5,
    'type_preprocess_insulin' : 'sum',      # 'sum'   'iob'
    
    # ===== monotonic gain loss =====
    'use_low_pass_I' : True,   # per ora implementato solo in strategy 1     !!!!!!
    
}
```


