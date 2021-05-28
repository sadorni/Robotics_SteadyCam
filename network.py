import pickle 
import struct
import socket
import numpy as np
import threading
import subprocess
import time


def send(s, data):
    data = pickle.dumps(data, protocol=2)
    s.sendall(struct.pack('>i', len(data)))
    s.sendall(data)

def recv(s): 
    data =  s.recv(4, socket.MSG_WAITALL)
    data_len = struct.unpack('>i', data)[0]
    data = s.recv(data_len, socket.MSG_WAITALL)
    return pickle.loads(data)

def listen_rpi():
    bashCommand = " ssh -t -t pi@192.168.1.118 'python server.py' "
    process = subprocess.run(bashCommand, shell=True)
    return

def Server():
    msg='Creating server on raspberry pi:'
    threading.Thread(target=lambda:[print(msg),listen_rpi()]).start()
    time.sleep(1)
    return

def Client():
    host = "192.168.1.118"
    port = 9395
    addr = (host, port)
    Server()
    while True:
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.connect(addr)
            print("Connection established")
            return(client)
            break
        except:
            print("Connection refused, trying again.")
            continue

def Communicate(position):
    client = Client()
    com={}
    com['send_rpi_data']=True
    com['coordinates'] = position
    send(client,com)
    while True: 
        return recv(client)
        if recv(client) == -1:
            break
    client.close()

