# BlindGrasp

## Installation
From the gym directory
```
 pip3 install -e .
```
## Task 
Random number of spheres and lego cubes are spawned into a tray.
The agent must learn to move the kuka robot towards the tray, explore the bottom of th tray and grasp the objects



### Discrete Action Space Environment
```
import gym
env=gym.make("gym_blg:blgd-v0", GUI=True)


```
NOTE
*  USE GUI=False when running on headless servers like google colab  or a remote server
*  If the agent learns by running multiple instances of env in parallel, use GUI=False

Action space is single dimensional and discrete which can have values from 0 to 13.
| Value     | Action    |
| ------------- |:-------------|
|0|Move Up 3 mm |
|1|Move Down 3mm|
|2|Jump Up 55mm |
|3|Jump Down 55 mm |
|4|Move Left 10mm |
|5|Move Right 10mm|
|6|Move Forward 10mm|
|7|Move Back 10mm|
|8|Move Front-Left 10mm|
|9|Move Front-Right 10mm|
|10|Move Back-Left 10mm|
|11|Move Back-Right 10mm|
|12|Open Gripper|
|13|Close Gripper |


### Continuous Environment
TODO


### Observation Space
Observation Space is a Tuple constsiting of five Box data elements. 
```
[proximity,position,force,maps,gelsight]
```
#### proximity
It is the reading of the proximity sensors (14 on the side + 8 at the tip). Its shape is (22,), dtype=uint8 and has values of 0 if no object and 1 if object in proximity.


#### position
It is the cartesian x,y,z coordinates. Positions history from the previous two timesteps are also appended to the data, to make use of temporal informations. Its shape is (9,), dtype=float64 and has normalized values between 0 and 1

#### force
Wrist force sensor readings. Shape (3,), dtype=float64. Normalized between 0 and 1

#### maps
shape=(32,32,2)
float64

#### gelsight
shape=(32,32,2)
float64
