import socket
import struct
import numpy as np

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Python pronto (32-bit mode). In attesa di Matlab...")
    conn, addr = s.accept()
    
    with conn:
        while True:
            # Riceviamo ad esempio una matrice 3x3 (9 numeri float32)
            # Ogni float32 occupa 4 byte. 9 * 4 = 36 byte.
            data = conn.recv(36) 
            if not data: break
            
            # Converte i byte in un array numpy float32
            matrice = np.frombuffer(data, dtype=np.float32).reshape((1, 2))
            print("Ricevuto:\n", matrice)
            
            # Esegui i conti
            risultato = (matrice * 10).astype(np.float32)
            
            # Rispedisci i byte a Matlab
            conn.sendall(risultato.tobytes())