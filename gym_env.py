import cv2
from IPython.display import clear_output
import time
import PIL.Image
from io import BytesIO
import IPython.display
import numpy as np
import math
import socket
import pickle 
import struct
#import listen
import gym
from gym import spaces
import cv2
from network import Communicate
#import threading



"""
The goal of the project is for the red dot on the raspberry pi to be as close as possible to the centre of the image captured from the webcam.

CONTINUOUS STATE:
We start from a random position of the red dot, the state of the dot is determined by the two coordinates on the led grid (x,y) and the 6 variables returned by the accelerometer and gyroscope of the raspberry pi (3 for each along the x,y,z axes). The latter are necessary to determine the relative position of the red dot with respect to space (where is up, where is down, etc...). 
ACTION:
There are 4 discrete actions: 
0: move right
1: move left
2: move up
3: move down

REWARD: 
We define the reward so that it is highest when the dot is closest to the centre of the image, this is computed using distance.We will use the following distribution as a reward: R = -distance


Reference: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
"""

N_ACTIONS = 4
N_ACC_VALUES = 3
N_GYRO_VALUES = 3
N_COORD = 2
HEIGHT = N_ACC_VALUES + N_GYRO_VALUES + N_COORD
WIDTH = N_ACC_VALUES + N_GYRO_VALUES + N_COORD

# Number of possible locations of the red dot on the Raspberry Pi (rp)
num_rp_x = 8
num_rp_y = 8


class SteadyCamEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SteadyCamEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Using discrete actions:
        self.action_space = spaces.Discrete(N_ACTIONS)
        # Using continuous observation space:
        self.observation_space = spaces.Box(low=0, high=320, shape=
                    (HEIGHT, WIDTH), dtype=np.float16)
        
        #Initiating timestep to 0
        #We will limit the number of timesteps to 100
        self._timestep = 0
        #Initiating reward to 0
        self.reward = -1000
        
        #Defining the maximum number of x and y positions
        self.max_rp_x = num_rp_x - 1
        self.max_rp_y = num_rp_y - 1


    def get_dot_coord(self):
        try:
            cap = cv2.VideoCapture(0) # Capture video from camera
            bkg=0
            #Had to introduce this to actually display the image, 
            #cv2 imshow doesn't display anything
            d = IPython.display.display("", display_id=1)
            time.sleep(1)
            # Capture frame-by-frame
            success, img = cap.read()
            #flip image for natural viewing
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (320, 320))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            #We work with little light since with the MAC webcam one cannot set the exposure, 
            #ISO properties, etc...
            #In this environment the following mask accurately finds the red dot
            #Other strategies, such as thresholds on grayscale images were tried but this proved effective
            
            mask = img[:,:,0]>254
            mask = mask.astype(np.uint8)
            
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            x, y = 50, 50
            max_area = None
            
            for stat, center in zip(stats[1:], centroids[1:]):
                area = stat[4]
                if (max_area is None) or (area > max_area):
                    x, y = center
                    max_area = area
            #Save position found
            x, y = int(x), int(y)
            img2 = np.copy(img)
            img2[y-10:y+10, x-10:x+10, :] = (100, 100, 255)
            #cv2.imshow('red_dot', img2)
            
            #Convert image to PIL image
            img_pil = PIL.Image.fromarray(img2)
            d.update(img_pil)
            #Compute distance
            w,h = img_pil.size
            x_ctr, y_ctr = w/2, h/2
            
            dist_x, dist_y = abs(x - x_ctr), abs(y-y_ctr)
            self.dist = math.sqrt(math.pow(dist_x,2) + math.pow(dist_y,2))
            return x,y
       
        except Abort:
            cap.release()
            IPython.display.clear_output()
            #cv2.destroyWindow('red_dot')

    def step(self, action):
        #Store everytime the step function is called 
        self._timestep +=1
        print(self._timestep)
        done = False
        #Open the socket and make the server listen
        #Implementing a threading so that this action doesn't block the running of the script while waiting 
        #for a response
        #a="Establishing connection:"
        #threading.Thread(target=lambda:[print(a),listen.Listen()]).start()
        #time.sleep(1)
        #Get new red dot location on the SenseHat pad based on the action 
        #But rp_x and rp_y are bounded between 0 and max_rp_x/max_rp_y
        if action == 0:
            #move right
            self.new_rp_x = min(self.rp_x + 1, self.max_rp_x)
            self.rp_x = self.new_rp_x
        elif action == 1:
            #move left
            self.new_rp_x = max(self.rp_x - 1, 0)
            self.rp_x = self.new_rp_x
        elif action == 2:
            #move down
            self.new_rp_y = min(self.rp_y + 1, self.max_rp_y)
            self.rp_y = self.new_rp_y
        elif action == 3:
            #move up
            self.new_rp_y = max(self.rp_y - 1, 0)
            self.rp_y = self.new_rp_y

        position = [self.rp_x, self.rp_y]
        print(position)

        #Put the red dot in (rp_x, rp_y) location on pixel pad 
        #And retrieve the dictionary containing accelerometer and gyroscopic data
        sense_data_dic = Communicate(position)
        red_dot_x, red_dot_y = self.get_dot_coord()
        state_coord = np.array([red_dot_x, red_dot_y])
        sense_data = sense_data_dic['sense_data']
        self.current_state = np.concatenate((state_coord, sense_data))
        #define reward
        self.reward = -1*self.dist
        print("Reward: " + str(self.reward))
        return self.current_state, self.reward, done or self._timestep>100, {}

    #Reset puts the red dot in position (0,0) on the SenseHat pad, timestep back to 0, reward back to 0
    #Get the current accelerometer and gyrometer data
    #And returns the current state containing these data and the red dot coordinates on the image
    #The agent then applies the first action on this reset state
    def reset(self):
        self.rp_x = 0
        self.rp_y = 0
        self._timestep = 0
        self.reward = -1000
        #a="Establishing connection:"
        #threading.Thread(target=lambda:[print(a),listen.Listen()]).start()
        #time.sleep(1)
        position = [self.rp_x, self.rp_y]
        sense_data_dic = Communicate(position)
        red_dot_x, red_dot_y = self.get_dot_coord()
        state_coord = np.array([red_dot_x, red_dot_y])
        sense_data = sense_data_dic['sense_data']
        self.current_state = np.concatenate((state_coord, sense_data))
        return self.current_state











