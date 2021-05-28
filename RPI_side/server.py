import socket
import pickle 
import struct
from utils.SenseFunc import Draw, get_gyro_data, get_acc_data, get_rpi_data

def send(s, data):
    data = pickle.dumps(data) 
    s.sendall(struct.pack('>i', len(data))) 
    s.sendall(data)

def recv(s):
    data = s.recv(4, socket.MSG_WAITALL) 
    data_len = struct.unpack('>i', data)[0] 
    data = s.recv(data_len, socket.MSG_WAITALL) 
    return pickle.loads(data)

# Create server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
s.bind(('0.0.0.0', 9395))
s.listen(10)
print('Waiting for a connection')
while True:
    # Accept a client
    conn, addr = s.accept()
    # conn is now a socket, if you write to it, the client receives that data. 
    # If you read from it, you get what the client sent you
    print("Connected to: ", addr)

    send_data = False

    try:
        com = recv(conn)
        if com:
            print(com)
            for key in com:
                if key=='send_rpi_data':
                    print(com[key])
                    send_data = com[key]
                if key=='coordinates':
                    vec=com[key]
                    X = vec[0]
                    Y = vec[1]
                    Draw(X,Y)
                if send_data:
                    send(conn,get_rpi_data())
        else:
            print("No more data")
    finally:
        send(conn, -1)
        break
conn.close()




