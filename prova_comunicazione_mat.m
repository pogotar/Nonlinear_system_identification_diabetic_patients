% Connessione
t = tcpclient('127.0.0.1', 65432, 'Timeout', 60);

for i = 1:10
    % Crea matrice 3x3 di tipo single (32-bit)
    % A = single(rand(3,3)); 
    A = single([12.123456789123456789, 11.987654321987654321]);
    
    % Invia la matrice come flusso di byte
    write(t, A(:)); % A(:) linearizza la matrice per l'invio
    
    fprintf('Inviata matrice 32-bit. In attesa...\n');
    
    % Leggi la risposta: sappiamo che aspettiamo 9 numeri single
    % 'single' in Matlab forza la lettura a 32-bit
    risposta_raw = read(t, 2, "single"); 
    
    % Ricostruisci la forma della matrice (3x3)
    matrice_python = reshape(risposta_raw, [1, 2]);
    
    disp('Risultato da Python:');
    disp(matrice_python);
    pause(2);  % 0.5
end

clear t;