import random


fasce = {
"mattina": (420, 540), # 07:00-09:00
"pomeriggio": (780, 900), # 13:00-15:00
"sera": (1200, 1320) # 20:00-22:00
}

giorni = 30
incremento_giorno = 1440 # minuti in un giorno
sequenza = []

minuti_cumulativi = 0

for _ in range(giorni):
    fascia = random.choice(list(fasce.keys()))
    inizio, fine = fasce[fascia]
    # Scegli un minuto casuale nella fascia
    minuti_giorno = random.randrange(inizio, fine+1, 5)
    # Somma all'incremento cumulativo
    minuti_cumulativi += incremento_giorno
    # Regola il valore alla fascia scegliendo l'orario relativo alla fascia nel giorno cumulativo
    minuti_finali = minuti_cumulativi - incremento_giorno + minuti_giorno
    sequenza.append(minuti_finali)

print(sequenza)