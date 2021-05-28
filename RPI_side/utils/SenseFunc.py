from sense_hat import SenseHat

def Draw(X,Y):
    sense = SenseHat()
    sense.clear()

    sense.set_pixel(X,Y,255,0,0)

def get_gyro_data():
    sense = SenseHat()
    return(sense.get_gyroscope_raw())

def get_acc_data():
    sense = SenseHat()
    return(sense.get_accelerometer_raw())

def get_rpi_data():
    gyro = get_gyro_data()
    acc = get_acc_data()
    dic = {}
    data = []
    data.append(gyro['x'])
    data.append(gyro['y'])
    data.append(gyro['z'])
    data.append(acc['x'])
    data.append(acc['y'])
    data.append(acc['z'])
    dic['sense_data']=data
    return(dic)




