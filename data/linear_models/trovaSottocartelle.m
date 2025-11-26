function sottocartelle_vettore = trovaSottocartelle(nome_cartella_principale)
% TROVASOTTOCARTELLEVETTORE Restituisce un vettore di celle contenente i
% nomi di tutte le sottocartelle (escludendo '.' e '..') di una cartella.

% 1. Ottiene il contenuto della cartella
contenuto_cartella = dir(nome_cartella_principale);

% 2. Inizializza un vettore di celle vuoto per memorizzare i nomi
nomi_sottocartelle = {};

% 3. Itera su tutti gli elementi trovati
for i = 1:length(contenuto_cartella)
    item = contenuto_cartella(i);

    % Verifica se è una directory E non è la directory corrente ('.')
    % o la directory padre ('..')
    if item.isdir && ~strcmp(item.name, '.') && ~strcmp(item.name, '..')
        % Aggiunge il nome al vettore di celle
        nomi_sottocartelle{end+1} = item.name;
    end
end

% 4. Assegna il risultato finale
sottocartelle_vettore = nomi_sottocartelle;

end

