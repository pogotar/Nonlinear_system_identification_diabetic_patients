%% Caricamento Parametri
data = load('ssm_all_params.mat');

% Esempio di segnale di input u [Time x D_in]
u = randn(100, size(data.ssm0.encoder_w, 2));


loaded = load("data.mat");

%% ESECUZIONE SSM 0
fprintf('Esecuzione SSM 0...\n');
y0 = simulate_ssm(loaded.u0_batch(1,:)', data.ssm0);

%% ESECUZIONE SSM 1
fprintf('Esecuzione SSM 1...\n');
y1 = simulate_ssm(loaded.u1_batch(1,:)', data.ssm1);

%% Confronto Risultati
figure;
subplot(3,1,1); plot(y0); title('Output SSM 0'); grid on;
subplot(3,1,2); plot(y1); title('Output SSM 1'); grid on;
subplot(3,1,3); plot(y0-y1); title('Output SSM 0-1'); grid on;


%% FUNZIONE DI SIMULAZIONE (Logica LRU + Gating LGLU)
function y_out = simulate_ssm(u_in, ssm_struct)
    T = size(u_in, 1);
    x = u_in * ssm_struct.encoder_w'; % Spazio d_model (8 canali)
    
    for i = 1:length(ssm_struct.blocks)
        b = ssm_struct.blocks{i};
        
        % --- Dinamica SSM (Block2x2) ---
        n_pairs = length(b.rho_raw); dx = n_pairs * 2;
        rho = 1 ./ (1 + exp(-b.rho_raw)) * (1 - 0.001); 
        K11 = zeros(dx);
        for p = 1:n_pairs
            idx = (p-1)*2 + (1:2);
            c = cos(b.theta(p)); s = sin(b.theta(p));
            K11(idx,idx) = rho(p) * [c -s; s c];
        end
        K_raw = [K11, b.K12; b.K21, b.K22];
        K = K_raw / (norm(K_raw, 2) + 0.002);
        
        gamma_val = exp(b.log_gamma);
        Az = K(1:dx, 1:dx); Bz = gamma_val * K(1:dx, dx+1:end);
        Cz = K(dx+1:end, 1:dx); Dz = gamma_val * K(dx+1:end, dx+1:end);
        
        % Ricorsione temporale
        z_state = zeros(dx, 1); y_lru = zeros(T, size(Cz,1));
        for t = 1:T
            ut = x(t,:)';
            y_lru(t,:) = (Cz * z_state + Dz * ut)';
            z_state = Az * z_state + Bz * ut;
        end
        
        % --- Gating LGLU (16 -> 8 canali) ---
        temp_ff = y_lru * b.ff_w' + b.ff_b;
        d_model = size(x, 2);
        val = temp_ff(:, 1:d_model);
        gate = temp_ff(:, d_model+1:end);
        z_ff = val .* (1 ./ (1 + exp(-gate))); % Sigmoid Gating [cite: 10, 16]
        
        % Connessione Residua
        x = x + z_ff;
    end
    y_out = x * ssm_struct.decoder_w';
end