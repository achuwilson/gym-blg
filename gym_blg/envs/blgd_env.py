import gym
import os
import sys

sys.path.append(os.path.dirname(__file__))

from gym import error, spaces, utils
from gym.spaces import Box, Tuple
from gym.utils import seeding
import numpy as np
import blg_core

#Distace delta values
XY_DIST=0.01 #1 cm
Z_DIST=0.003 #3mm
Z_JUMP=0.055 #5cm
#xpositive, xnegative etc
xp=np.array([XY_DIST,0,0,0])
xn=np.array([-XY_DIST,0,0,0])
yp=np.array([0,XY_DIST,0,0])
yn=np.array([0,-XY_DIST,0,0])
zp=np.array([0,0,Z_DIST,0])
zn=np.array([0,0,-Z_DIST,0])
gro=np.array([0,0,0,-1])
grc=np.array([0,0,0,1])
zpp=np.array([0,0,Z_JUMP,0])
znn=np.array([0,0,-Z_JUMP,0])


xpyp=np.array([XY_DIST,XY_DIST,0,0])
xpyn=np.array([XY_DIST,-XY_DIST,0,0])
xnyp=np.array([-XY_DIST,XY_DIST,0,0])
xnyn=np.array([-XY_DIST,-XY_DIST,0,0])

def move_up(b=0):
    return(zp)

def move_down(b=0):
    return(zn)
    
def move_up_j(b=0):
    return(zpp)
def move_down_j(b=0):
    return(znn)
    
def move_l(b=0):
    return(yn)
def move_r(b=0):
    return(yp)
def move_f(b=0):
    return(xp)
def move_b(b=0):
    return(xn)
    
def move_fl(b=0):
    return(xpyn)
def move_fr(b=0):
    return(xpyp)
def move_bl(b=0):
    return(xnyn)
def move_br(b=0):
    return(xnyp)
    
def move_gro(b=0):
    return(gro)
def move_grc(b=0):
    return(grc)

funcDict={0:move_up,1:move_down,2:move_up_j,3:move_down_j,4:move_l,5:move_r,6:move_f,7:move_b,8:move_fl,9:move_fr,10:move_bl,11:move_br,12:move_gro,13:move_grc}

class BlgDiscreteEnv(gym.Env):
    metadata = {'render.modes': ['human']}
 
    def __init__(self, GUI=False, ResetCount=2500):
        
        #GUI=True
        #LAPTOP_MODE=True
        #GPU=False

        self.blg=blg_core.BlindGrasp(GUI_MODE=GUI)
        self.blg.reset()
        self.action_space = spaces.Discrete(14)
        self.blg.MaxSteps = ResetCount

        #observation space parameters
        '''
        prox_low=np.zeros(14)
        prox_high=np.ones(14)
        pos_low=np.array([self.blg.X_MIN-0.01,self.blg.Y_MIN-0.01,self.blg.Z_MIN,self.blg.X_MIN-0.01,self.blg.Y_MIN-0.01,self.blg.Z_MIN,self.blg.X_MIN-0.01,self.blg.Y_MIN-0.01,self.blg.Z_MIN])
        pos_high=np.array([self.blg.X_MAX+0.01,self.blg.Y_MAX+0.01,self.blg.Z_MAX+0.06,self.blg.X_MAX+0.01,self.blg.Y_MAX+0.01,self.blg.Z_MAX+0.06,self.blg.X_MAX+0.01,self.blg.Y_MAX+0.01,self.blg.Z_MAX+0.06])
        force_low=np.array([-7.0,-7.0,-7.0])
        force_high=np.array([7,7,7])
        maps_space=Box(low=0,high=1,shape=(32,32,2),dtype=np.float32)
        gel_space=Box(low=0,high=1,shape=(32,32,2),dtype=np.float32)
        prox_space=Box(low=prox_low,high=prox_high,dtype=np.uint8)
        pos_space=Box(low=pos_low,high=pos_high,dtype=np.float32)
        force_space=Box(low=force_low,high=force_high,dtype=np.float32)
        self.observation_space=Tuple([prox_space,pos_space,force_space,maps_space,gel_space])
        '''
        #prox=22, pos=9,force=3, maps =3*32*32, gelsight 2*32*32 => total=5154
        self.observation_space = Box(low=0,high=1,shape=(5154,))

        

 
    def step(self, action):
        
        reward,Done, obs=self.blg.step(funcDict[action]())
        '''
        prox=obs[0] #proximity data, min0, max 1
        pos=obs[1]
        pos=pos.reshape(3,3)-np.array([self.blg.X_MIN,self.blg.Y_MIN,0])# subtract 0.416 from x pos, add 0.187 to y
        pos=pos.reshape(9,)
        force=obs[2]
        force=np.clip(force, -7,7)
        force=np.interp(force, (-7, 7), (0, +1))
        maps=np.stack((obs[3],obs[4]),axis=2)
        maps= maps/3.0
        gelsight=np.stack((obs[5],obs[6]),axis=2)
        gelsight=gelsight/255.0
        step_obs= np.array([prox,pos,force,maps,gelsight])
        '''

        prox=obs[0] #proximity data, min0, max 1
        pos=obs[1]
        pos=pos.reshape(3,3)-np.array([self.blg.X_MIN,self.blg.Y_MIN,0])# subtract 0.416 from x pos, add 0.187 to y
        pos=pos.reshape(9,)
        force=obs[2]
        force=np.clip(force, -7,7)
        force=np.interp(force, (-7, 7), (0, +1))
        
        '''
        maps=np.stack((obs[3],obs[4]),axis=2)
        maps= maps/3.0
        gelsight=np.stack((obs[5],obs[6]),axis=2)
        gelsight=gelsight/255.0
        return np.array([prox,pos,force,maps,gelsight])
        '''
        objmap=obs[3].flatten()
        visitmap=obs[4].flatten()
        curmap = obs[5].flatten()

        gelsight1=obs[6].flatten()
        gelsight2=obs[7].flatten()
        step_obs = np.concatenate((prox,pos,force,objmap,visitmap,curmap,gelsight1,gelsight2))

        return step_obs,reward,Done, {}

    
    def reset(self):
        self.blg.reset()
        #just make a small move down to fix the bug of the initial negative reward
        self.step(1)
        return self._get_obs()
        
    
    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        obs=self.blg.getObservation()
        #normalize  observations
        prox=obs[0] #proximity data, min0, max 1
        pos=obs[1]
        pos=pos.reshape(3,3)-np.array([self.blg.X_MIN,self.blg.Y_MIN,0])# subtract 0.416 from x pos, add 0.187 to y
        pos=pos.reshape(9,)
        force=obs[2]
        force=np.clip(force, -7,7)
        force=np.interp(force, (-7, 7), (0, +1))
        
        '''
        maps=np.stack((obs[3],obs[4]),axis=2)
        maps= maps/3.0
        gelsight=np.stack((obs[5],obs[6]),axis=2)
        gelsight=gelsight/255.0
        return np.array([prox,pos,force,maps,gelsight])
        '''
        objmap=obs[3].flatten()
        visitmap=obs[4].flatten()
        curmap = obs[5].flatten()

        gelsight1=obs[6].flatten()
        gelsight2=obs[7].flatten()
        obs = np.concatenate((prox,pos,force,objmap,visitmap,curmap,gelsight1,gelsight2))
        return obs
        


