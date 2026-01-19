

modifiche.Stipendio = {5, 0.1924}; % {3, 2500; 7, 3000}
modifica_xlsx_per_nome_colonna('results.xlsx', modifiche)




function modifica_xlsx_per_nome_colonna(file_path, modifiche)
    % Modifica valori cercando la colonna per NOME (dalla prima riga).
    %
    % Input:
    %   file_path: percorso del file .xlsx (es: 'dati.xlsx')
    %   modifiche: struct dove ogni campo è il nome della colonna
    %             e contiene un array di celle da modificare
    %
    % Esempio:
    %   modifiche.Nome = {2, 'Marco'; 5, 'Luca'};
    %   modifiche.Eta = {2, 30; 5, 25; 8, 28};
    %   modifiche.Stipendio = {3, 2500; 7, 3000};
    %   modifica_xlsx_per_nome_colonna('dati.xlsx', modifiche)
    
    excel = actxserver('Excel.Application');
    excel.Visible = false;
    
    try
        % Apri il workbook
        workbook = excel.Workbooks.Open(fullfile(pwd, file_path));
        worksheet = workbook.Sheets.Item(1);
        
        % Leggi la prima riga (intestazione)
        intestazione_range = worksheet.Range('1:1');
        intestazione = intestazione_range.Value;
        
        % Crea un map: nome_colonna -> indice_colonna
        col_map = containers.Map();
        for col_idx = 1:length(intestazione)
            nome_col = intestazione{col_idx};
            if ~isempty(nome_col)
                col_map(nome_col) = col_idx;
            end
        end
        
        % Modifica le celle
        nomi_colonne = fieldnames(modifiche);
        for i = 1:length(nomi_colonne)
            nome_colonna = nomi_colonne{i};
            
            if ~isKey(col_map, nome_colonna)
                fprintf('⚠️ Colonna ''%s'' non trovata!\n', nome_colonna);
                continue;
            end
            
            col_idx = col_map(nome_colonna);
            righe_valori = modifiche.(nome_colonna);
            
            for j = 1:size(righe_valori, 1)
                riga = righe_valori{j, 1};
                valore = righe_valori{j, 2};
                
                cella_ref = sprintf('%s%d', colnum2colstr(col_idx), riga);
                worksheet.Range(cella_ref).Value = valore;
                fprintf('Modificato %s (riga %d): %s\n', nome_colonna, riga, mat2str(valore));
            end
        end
        
        % Salva il file
        workbook.Save();
        fprintf('\n✓ File %s aggiornato con successo\n', file_path);
        
    catch ME
        fprintf('Errore: %s\n', ME.message);
    finally
        workbook.Close(0);
        excel.Quit();
        delete(excel);
    end
end

% Funzione helper: converte numero colonna in lettera (es: 1 -> 'A', 27 -> 'AA')
function col_str = colnum2colstr(col_num)
    col_str = '';
    while col_num > 0
        col_num = col_num - 1;
        col_str = char(65 + mod(col_num, 26)) + col_str;
        col_num = floor(col_num / 26);
    end
end