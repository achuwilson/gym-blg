# BlindGrasp



## **Local Installation**

#### **Create Virtual Env**

Create a python virtual env
```
python3 -m venv blgenv
```
Activate it
```
 source blgenv/bin/activate
 ```
Install Ipython for Jupyter notebook
```
pip install ipykernel
```
Add the venv to Jupyter
```
python -m ipykernel install --user --name=blgenv
```
also for the GUI
```
pip install ipywidgets
```
#### **Install the BlindGrasp gym environmet**

From the gym-blg directory, install the gym environment
```
 pip3 install -e .
```
Note: if the error ```error: invalid command 'bdist_wheel'``` shows up, install wheel: ``` pip3 install wheel```

#### **Install Tensorflow and Stable Baselines**
Install Tensorflow 1.14 ( Stable baselines support only TF1 as of now)
```
pip install tensorflow-gpu==1.14
```
Install Stable Baselines
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

pip install stable-baselines[mpi]

```
Note: if it throws an error with skbuild not found, install it :  ```pip install scikit-build```

Note: opencv-python installation associated with stable-baselines sometimes takes a long time. 



## **Installation on Google Colab**
See this Google Colab [Notebook](https://colab.research.google.com/drive/1Sgng1H7ZC21iQxenNCkiZeAFjbccyJYV?usp=sharing)

Note: GUI not available with Colab


# **Task 1**
Random number of spheres and lego cubes are spawned into a tray.
The agent must learn to move the kuka robot towards the tray, explore the bottom of th tray and grasp the objects



### **Discrete Action Space Environment**
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


### **Continuous Environment**
TODO


### **Observation Space**
Observation Space is OpenAI gym Box type with 5154 elements. 
It consists of normalized sensor data from proximity sensors, end effector position, force sensors, map of visited area and two gelsight sensors. 

Refer to `decode_obs()` function in `agents/human_demonstrations.ipynb` for details


