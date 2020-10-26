#!/usr/bin/python3

import pybullet 
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np
import math
import time
import random
import cv2 #optional
import os
import sys
curPath=os.path.dirname(__file__)
sys.path.append(curPath)


class BlindGrasp:
    FREQ=240.0
    TRAY_X=0.6
    TRAY_Y=0.0
    TRAY_LEN=0.41
    LEGO_TRAY_X = 0.3
    LEGO_TRAY_Y=0.5
    SPHERE_TRAY_X=0.3
    SPHERE_TRAY_Y=-0.5

    CYL_RAD=0.002
    CYL_HEIGHT=0.0025
    CYL_MASS=0.0000001
    SENS_XOFF=0.052 #zero offset between the sensor array centre and geometric origin
    SENS_RAD=0.032 # radius of the sensor array origin 
    SENS_ANGLES=[0.0,30.0,60.0,82.5,97.5,120.0,150.0,180.0,210.0,240.0,262.5,277.5,300.0,330.0]
    
    SENS_HEIGHTS=[0.09, 0.07,0.05,0.03]
    PROX_SENS_RANGE=0.01
    PROX_SENS_RANGE_TIP=0.02
   
    SENS_LAYERS=1 #number of layers used, ie from the tip

    #TIP SENSOR POSITIONS
    TIP_SENSE_Z=0.1035#0.1025
    TIP_SENSE_Y=0.02
    TIP_SENSE_X=0.055
    TIP_SENSE_XM1=0.055
    TIP_SENSE_XM2=0.072

    #CAMERA PARAMS FOR GELSIGHT
    # 0.061=0.02(base thickness) + 0.08/2 (mid of finger height) 
    #       edit - move z 5mm up to correct gelsight image =>0.066
    #0.0745=0.0495(offset) + 0.025 ( half thickness of dodecahedron ie 50mm/2)
    L_CAM_OFF=np.array([-0.0745,0,0.066])
    CAM_FOCAL=0.025 
    L_CAM_VEC=[CAM_FOCAL,0,0]
    l_init_up_vector = (0, 1, 0) #Camera axis

    R_CAM_OFF=np.array([0.0745,0,0.066])
    R_CAM_VEC=[-CAM_FOCAL,0,0]
    r_init_up_vector = (0, -1, 0) #Camera axis
            #Gelsight camera params
    camFOV=75.0
    camNearVal=0.02 #min distance to sense
    camFarVal=0.04 #max diatance to sense

    ProxMap=[[2,5],[3,6],[4,7],[5,8],[6,8],[7,7],[8,6],[9,5],[8,4],[7,3],[6,2],[5,2],[4,3],[3,4],[0,5],[2,7],[4,9],[5,10],[6,10],[7,9],[9,7],
    [11,5],[9,3],[7,1],[6,0],[5,0],[4,1],[2,3],[6,6],[6,5],[7,5],[6,4],[5,4],[5,5],[4,5],[5,6]]

    GRASP_FTHRES=0.1 #threshold force for grasp
    # The height to which the ee should move down, to explore      
    #if z  below 0.172, lego grasp fails
    #if z  above 0.2, sphere and lego grasp fails
    #so, ((0.2-0.172)/2)+0.172 = 0.186
    GripperTrayZ=0.182 
    deltaPoseZ=0.006 #distance dalta for z movement
    deltaPoseZSmall=0.0025 # used on first approach, to place it midway
    deltaPoseZHigh =0.056 # penalize if we go above this 
    ExtractZ=0.35
    scaleRZ = 20
    scaleRZLow=0.3
    startX=0.6
    startY=0.0
    startZ=0.35
    prevZ=startZ
    deltaPrevZ=0.002

    # TRAY CONTACT by gripper
    #y axis contact at 0.185 & -0.186
    #x axis- gripper open -  0.464 & 0.736
    #x axis- gripper closed 0.417 &0.786
    X_MIN=0.417
    X_MAX=0.786
    Y_MIN=-0.186
    Y_MAX=0.185
    Z_MIN = 0.173
    Z_MAX= 0.36

    scaleRXY =800
    TrayContactPenalty = -100
    ObjContactPenalty = -2
    HighObjContactPenalty = -10
    safeLoadPos_off=0.05 #safe offset when loading objs, in x axis, so that object is placed such that gripper can pick it up without traycontact
    MapDeltaReward =  5

    GripReward =1000
    DoneReward = 10000
    ReachZUpReward =10
    ReachedZTrayReward=30
    MapCoverReward =100

    StepPenalty = -1 
    DownMoveReward = 5
    ZContactPenalty = -5   #z axis collision penalty
    ZFthres = 0 # force threshold for z axis contact penalty

    #MAP
    # the corners of tray are (0.395,0.205)(0.805,0.205)(0.805,-0.205)(0.395,-0.205)
    # The are can be divided into 32x32 squares of 1.28125 cm side length
    MAP_COLS=32
    MAP_ROWS=32

    #max number of random objects generated to pick up
    MAX_OBJ_NUM=10

    #vertical gripper orientation for picking up
    pose_down=[0,180,0]
    ORN_DOWN=pybullet.getQuaternionFromEuler([math.radians(pose_down[0]),math.radians(pose_down[1]),math.radians(pose_down[2])])

    def __init__(self, GUI_MODE):
        #to keep track of grasp status
        self.GraspFlag=False
        self.GraspCommanded = False
        self.prevGripCmd=False # if the previous step was gripper close

        self.kuka=None
        self.blgripper=None
        self.legotray=None
        self.spheretray=None
        self.posStack=None

        self.num_joints_k=None
        self.num_joints_g=None
        self.joint_dict = None
        self.up_limits = None
        self.low_limits = None
        self.VisitMap = np.zeros((self.MAP_ROWS,self.MAP_COLS),dtype = np.uint8) #for every visited tile, the corresponding matrix entry is marked 1 
        self.ObjMap =np.zeros((self.MAP_ROWS,self.MAP_COLS),dtype = np.uint8) #the map which marks object positions
        self.CurPosMap =  np.zeros((self.MAP_ROWS,self.MAP_COLS),dtype = np.uint8) #map which shows the current position of end effector
        self.ProxDataMap=np.zeros((12,11),dtype = np.uint8) #just to show the proxy sensor data in human understandable form

        #to keep track of area explored 
        self.prevMapArea=0
        self.areaCovered=0

        self.previousGraspFlag = False
        self.reachedTrayFlag=False
        self.checkDropped = True
        self.GraspFlag = False
        self.objInHandFlag=False
        self.GraspCount=0
        self.DoneFlag=False
        
        self.StepCount=0
        self.legoIds=[]
        self.sphereIds=[]
        #generate number of objects
        self.totalObjs=None
        self.numSpheres=None
        self.numLegos=None
        self.SENS_POS=[]
        self.SENS_ORN=[]
        self.SENS_ID=[]
        self.GUI=False

        #maximum number of steps in an episode
        self.MaxSteps = 2500


        #init pybullet connection
        if(GUI_MODE==True):
            self.GUI=True
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
            self.p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            self.p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0.6,0,0.27])
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.p.resetSimulation()
        self.p.setRealTimeSimulation(0)
        self.p.setGravity(0,0,-10.0)

    def loadRobot(self):
        self.kuka = self.p.loadURDF('kuka_iiwa/model.urdf', useFixedBase = 1)
        self.blgripper=self.p.loadURDF(curPath+"/data/blgripper.urdf",[0,0,1.35]) 
        self.tray=self.p.loadURDF(curPath+"/data/tray/tray.urdf",[self.TRAY_X,self.TRAY_Y,0.0],useFixedBase = 1)
        self.legotray=self.p.loadURDF(curPath+"/data/tray/tray.urdf",[self.LEGO_TRAY_X,self.LEGO_TRAY_Y,0.02],useFixedBase = 1,globalScaling=0.5)
        self.spheretray=self.p.loadURDF(curPath+"/data/tray/tray.urdf",[self.SPHERE_TRAY_X,self.SPHERE_TRAY_Y,0.02],useFixedBase = 1,globalScaling=0.5)

        #attach gripper to robot
        blgripper_cid = self.p.createConstraint(self.kuka,6,self.blgripper,-1,pybullet.JOINT_FIXED, [0,0,0], [0,0,0.025],[0,0,0])
        self.num_joints_k = self.p.getNumJoints(self.kuka)
        self.num_joints_g = self.p.getNumJoints(self.blgripper)

        #create the prox sensors
        colCylId = self.p.createCollisionShape(pybullet.GEOM_CYLINDER,radius=self.CYL_RAD,height=self.CYL_HEIGHT)
        visCylId =self.p.createVisualShape(pybullet.GEOM_CYLINDER,radius=self.CYL_RAD,length=self.CYL_HEIGHT,rgbaColor=[1,1,0,1],specularColor=[1,1,1])
    
        #reset the sensor id, pos, orn
        self.SENS_ID=[]
        self.SENS_POS=[]
        self.SENS_ORN=[]

        for height in self.SENS_HEIGHTS:
            for angle in self.SENS_ANGLES:
                if(math.cos(math.radians(angle))>0.0): #if positive quadrants of x, add
                    self.SENS_POS.append([self.SENS_XOFF+self.SENS_RAD*math.cos(math.radians(angle)),self.SENS_RAD*math.sin(math.radians(angle)), height])
                else:
                    self.SENS_POS.append([-self.SENS_XOFF+self.SENS_RAD*math.cos(math.radians(angle)),self.SENS_RAD*math.sin(math.radians(angle)), height])
                self.SENS_ORN.append(self.p.getQuaternionFromEuler([math.radians(0),math.radians(90),math.radians(angle)]))
       
        #ADD TIP SENSORS
        #TIP LEFT POSITIONS - numbered counterclockwise when looking from top
        self.SENS_POS.append([-self.TIP_SENSE_X,self.TIP_SENSE_Y,self.TIP_SENSE_Z])
        self.SENS_POS.append([-self.TIP_SENSE_XM1,0,self.TIP_SENSE_Z])
        self.SENS_POS.append([-self.TIP_SENSE_XM2,0,self.TIP_SENSE_Z])
        self.SENS_POS.append([-self.TIP_SENSE_X,-self.TIP_SENSE_Y,self.TIP_SENSE_Z])
        #TIP RIGHT POSITIONS
        self.SENS_POS.append([self.TIP_SENSE_X,-self.TIP_SENSE_Y,self.TIP_SENSE_Z])
        self.SENS_POS.append([self.TIP_SENSE_XM1,0,self.TIP_SENSE_Z])
        self.SENS_POS.append([self.TIP_SENSE_XM2,0,self.TIP_SENSE_Z])
        self.SENS_POS.append([self.TIP_SENSE_X,self.TIP_SENSE_Y,self.TIP_SENSE_Z])
        #TIP SENSOR ORIENTATIONS
        for i in range(8):
            self.SENS_ORN.append(pybullet.getQuaternionFromEuler([math.radians(0),math.radians(0),math.radians(0)])) 

        #generate unique IDs for the sensors
        for i in range(len(self.SENS_POS)):
            uid= self.p.createMultiBody(self.CYL_MASS,colCylId,visCylId,[0,0,0.16])
            self.p.setCollisionFilterGroupMask(uid, 1, 0,0)
            self.p.setCollisionFilterGroupMask(uid, 0, 0,0)
            self.p.setCollisionFilterGroupMask(uid, -1, 0,0)
            self.SENS_ID.append(uid)
            if(self.SENS_POS[i][0]>0): #if positive x axis
                cyl_cid=self.p.createConstraint(self.blgripper,1,uid, -1,pybullet.JOINT_FIXED, [0,0,0], self.SENS_POS[i],[0,0,0], parentFrameOrientation=self.SENS_ORN[i])
            else: #if negative x axis
                cyl_cid=self.p.createConstraint(self.blgripper,0,uid, -1,pybullet.JOINT_FIXED, [0,0,0], self.SENS_POS[i],[0,0,0], parentFrameOrientation=self.SENS_ORN[i])       

        # Make an empty dictionary
        self.joint_dict = {}
        # Define a list of upper and lower limits
        self.up_limits = []
        self.low_limits = []
        # Fill the dict
        for i in range(self.num_joints_k):
            self.p.changeDynamics(self.kuka, i, linearDamping=0, angularDamping=0)
            # Get a list of joint information
            joint_info = self.p.getJointInfo(self.kuka, i)
            self.joint_dict.update({str(joint_info[1], 'utf-8') : joint_info[0]})
            # Get the limits
            self.up_limits.append(joint_info[9]), self.low_limits.append(joint_info[8])


        #enable force sensing on the two gripper axes and on kuka wrist
        self.p.enableJointForceTorqueSensor(self.blgripper,0,1)
        self.p.enableJointForceTorqueSensor(self.blgripper,1,1)
        self.p.enableJointForceTorqueSensor(self.kuka,self.num_joints_k-1,1)

        #wait for everything to settle down
        for i in range(400):
            self.p.stepSimulation()
            time.sleep(1.0/self.FREQ) 

        #open the gripper    
        self.openGripper()     


    def loadObjs(self):
        #load spheres and legos at random positions into the tray
        self.legoIds=[]
        self.sphereIds=[]

        #generate number of objects
        self.totalObjs=random.randint(1,self.MAX_OBJ_NUM)
        self.numSpheres=random.randint(1,self.totalObjs)
        self.numLegos=self.totalObjs-self.numSpheres

        #generate random position for objects - in x axis safety offset of 5 cm
        
        obPosX=np.random.uniform(((self.TRAY_X-(self.TRAY_LEN/2.0))+self.safeLoadPos_off),((self.TRAY_X+(self.TRAY_LEN/2.0))-self.safeLoadPos_off), self.totalObjs)
        obPosY=np.random.uniform((self.TRAY_Y-(self.TRAY_LEN/2.0)),(self.TRAY_Y+(self.TRAY_LEN/2.0)), self.totalObjs)
        obPosZ =0.18 #drop it from 0.1 m

        for num in range(self.numSpheres):
            #print("S",obPosX,obPosY)
            sphereId=self.p.loadURDF(curPath+"/data/sphere/sphere_small.urdf",[obPosX[num],obPosY[num],obPosZ],useFixedBase = 0)
            self.sphereIds.append(sphereId)
        for num in range(self.numSpheres,self.totalObjs):
            #print("L",obPosX[num],obPosY[num])
            legoId=self.p.loadURDF(curPath+"/data/legoV2/legoV2.urdf",[obPosX[num],obPosY[num],obPosZ],useFixedBase = 0)
            self.legoIds.append(legoId) 

    def removeAll(self):
        #remove the spheres and legos in the environment
        self.legoIds=[]
        self.sphereIds=[]  
        self.p.resetSimulation()

    def removeObjs(self):
        #remove the spheres and legos in the environment  
        for num in range(self.numSpheres):
            self.p.removeBody(self.sphereIds[num])
        for num in range(self.numLegos):
            self.p.removeBody(self.legoIds[num])
        self.legoIds=[]
        self.sphereIds=[]  

    def getGelSightData(self):
        #left finger is the finger in -ve x axis ( left side when looking from positive y axis, also finLeft.STL)
        # the origin of left finger is at 0, ie, the offset is positive x
        # right finger is on the +ve x axis( right side when looking from positive y axis, also finRight.STL)  
        # the origin of right finger is at 0 ie  the offset is towards -ve direction of x axis  

        #get the position of the origin of two fingers
        flpos=self.p.getLinkState(self.blgripper,0,computeForwardKinematics=True)
        frpos=self.p.getLinkState(self.blgripper,1,computeForwardKinematics=True)

        #Estimate the position of the Left finger Camera
        spos_l=np.array(flpos[0])+self.L_CAM_OFF
        spos_ =pybullet.multiplyTransforms(flpos[0],flpos[1],self.L_CAM_OFF,(0,0,0,1))
        spos_lcam=spos_[0] #Position of the Left Camera
        #Estimate the camera target position to look to
        lcampos_target=pybullet.multiplyTransforms(spos_lcam,flpos[1],self.L_CAM_VEC,[0,0,0,1])
        #Visualize
        #self.p.addUserDebugLine(spos_lcam,lcampos_target[0],[0,1,0],lineWidth=10, lifeTime=0.05)
        #More Visualization
        #lcampos2=pybullet.multiplyTransforms(spos_lcam,flpos[1],[0,0.3,0],[0,0,0,1])
        #lcampos3=pybullet.multiplyTransforms(spos_lcam,flpos[1],[0,0,0.3],[0,0,0,1])  
        #self.p.addUserDebugLine(spos_lcam,lcampos2[0],[0,0,1],lineWidth=10, lifeTime=0.05)
        #self.p.addUserDebugLine(spos_lcam,lcampos3[0],[1,1,0],lineWidth=10, lifeTime=0.05)

        #Estimate the position of the Right finger Camera
        spos_r=np.array(frpos[0])+self.R_CAM_OFF
        spos__=pybullet.multiplyTransforms(frpos[0],frpos[1],self.R_CAM_OFF,(0,0,0,1))
        spos_rcam=spos__[0] #position of the right finger camera
        #Estimate the camera target position to look to
        rcampos_target=pybullet.multiplyTransforms(spos_rcam,frpos[1],self.R_CAM_VEC,[0,0,0,1])
        #Visualize
        #self.p.addUserDebugLine(spos_rcam,rcampos_target[0],[0,1,0],lineWidth=10, lifeTime=0.05)
        #More visualisation
        #rcampos2=pybullet.multiplyTransforms(spos_rcam,frpos[1],[0,0.3,0],[0,0,0,1])
        #rcampos3=pybullet.multiplyTransforms(spos_rcam,frpos[1],[0,0,0.3],[0,0,0,1])
        #self.p.addUserDebugLine(spos_rcam,rcampos2[0],[0,0,1],lineWidth=10, lifeTime=0.05)
        #self.p.addUserDebugLine(spos_rcam,rcampos3[0],[1,1,0],lineWidth=10, lifeTime=0.05) 

        #compute Left camera image
        l_rot_matrix=pybullet.getMatrixFromQuaternion(flpos[1])
        l_rot_matrix = np.array(l_rot_matrix).reshape(3, 3)
        
        l_up_vector = l_rot_matrix.dot(self.l_init_up_vector)
        l_viewMatrix=pybullet.computeViewMatrix( cameraEyePosition=spos_lcam, cameraTargetPosition=lcampos_target[0],cameraUpVector=l_up_vector) 
        l_projectionMatrix = pybullet.computeProjectionMatrixFOV(fov=self.camFOV,aspect=1.0,nearVal=self.camNearVal,farVal=self.camFarVal)
        if(self.GUI):
            width, height, l_rgbImg, l_depthImg, l_segImg = self.p.getCameraImage(width=320,height=240,viewMatrix=l_viewMatrix,    projectionMatrix=l_projectionMatrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        else:
            width, height, l_rgbImg, l_depthImg, l_segImg = self.p.getCameraImage(width=320,height=240,viewMatrix=l_viewMatrix,    projectionMatrix=l_projectionMatrix,renderer=pybullet.ER_TINY_RENDERER)    

        #compute Right camera image
        r_rot_matrix=pybullet.getMatrixFromQuaternion(frpos[1])
        r_rot_matrix = np.array(r_rot_matrix).reshape(3, 3)
        
        r_up_vector = r_rot_matrix.dot(self.r_init_up_vector)
        r_viewMatrix=pybullet.computeViewMatrix( cameraEyePosition=spos_rcam, cameraTargetPosition=rcampos_target[0],cameraUpVector=r_up_vector) 
        r_projectionMatrix = pybullet.computeProjectionMatrixFOV(fov=self.camFOV,aspect=1.0,nearVal=self.camNearVal,farVal=self.camFarVal)
        if(self.GUI):
            width, height, r_rgbImg, r_depthImg, r_segImg = self.p.getCameraImage(width=320,height=240,viewMatrix=r_viewMatrix,    projectionMatrix=r_projectionMatrix,renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        else:
            width, height, r_rgbImg, r_depthImg, r_segImg = self.p.getCameraImage(width=320,height=240,viewMatrix=r_viewMatrix,    projectionMatrix=r_projectionMatrix,renderer=pybullet.ER_TINY_RENDERER)        
        #img=cv2.cvtColor(r_rgbImg,cv2.COLOR_BGRA2RGB) # if you want to convert to opencv image
        #invert the image
        #print("MAX",np.max(l_depthImg), np.max(r_depthImg))
        return(1.0-l_depthImg,1.0-r_depthImg)
       
    def getProximityData(self):
        rayStartPos=[]
        rayEndPos=[]
        #for i in range(len(SENS_POS)):
        for i in range((self.SENS_LAYERS*len(self.SENS_ANGLES))):
            spos=self.p.getBasePositionAndOrientation(self.SENS_ID[i])
            rayStartPos.append(spos[0])
            epos=self.p.multiplyTransforms(spos[0],spos[1],(0,0,self.PROX_SENS_RANGE),[0,0,0,1])
            rayEndPos.append(epos[0])
            if(self.GUI):
                self.p.addUserDebugLine(spos[0],epos[0],[1,0,0], lifeTime=0.05)

        for i in range((len(self.SENS_POS)-8),len(self.SENS_POS)):
        #for i in range((SENS_LAYERS*len(SENS_ANGLES)), ((SENS_LAYERS*len(SENS_ANGLES))+8)):
            spos=self.p.getBasePositionAndOrientation(self.SENS_ID[i])
            rayStartPos.append(spos[0])
            epos=self.p.multiplyTransforms(spos[0],spos[1],(0,0,self.PROX_SENS_RANGE_TIP),[0,0,0,1])
            rayEndPos.append(epos[0])
            if(self.GUI):
                self.p.addUserDebugLine(spos[0],epos[0],[1,0,0], lifeTime=0.05)

        rayTestResult=self.p.rayTestBatch(rayStartPos,rayEndPos)
        rayResult=[]
        for val in rayTestResult:
            if(val[0]>-1):
                rayResult.append(1)
            else:
                rayResult.append(0)    

        # this is just for human friendly visualization  
        if(self.GUI):
            self.ProxDataMap=np.zeros((12,11),dtype = np.uint8)  
            #get the readings from the lower layer       
            for i in range(len(self.SENS_ANGLES)): 
                self.ProxDataMap[self.ProxMap[i][0],self.ProxMap[i][1]]= (rayResult[i]) *1
            #get the tip sensor readings
            self.ProxDataMap[self.ProxMap[28][0],self.ProxMap[28][1]] = (rayResult[-8]) * 3
            self.ProxDataMap[self.ProxMap[29][0],self.ProxMap[29][1]] = (rayResult[-7]) * 3
            self.ProxDataMap[self.ProxMap[30][0],self.ProxMap[30][1]] = (rayResult[-6])* 3
            self.ProxDataMap[self.ProxMap[31][0],self.ProxMap[31][1]] = (rayResult[-5]) * 3
            self.ProxDataMap[self.ProxMap[32][0],self.ProxMap[32][1]] = (rayResult[-4]) * 3
            self.ProxDataMap[self.ProxMap[33][0],self.ProxMap[33][1]] = (rayResult[-3]) * 3
            self.ProxDataMap[self.ProxMap[34][0],self.ProxMap[34][1]] = (rayResult[-2]) * 3
            self.ProxDataMap[self.ProxMap[35][0],self.ProxMap[35][1]] = (rayResult[-1]) * 3   
    
        return(rayResult)

    def drawline(self):
        if(self.GUI):
            prox=self.getProximityData()   
            l_depth,r_depth=self.getGelSightData()
            l_depth=l_depth[0:240,80:320]
            r_depth=r_depth[0:240,80:320]
            
            cv2.imshow("L_DEPTH",l_depth)
            cv2.imshow("R_DEPTH",r_depth)
            cv2.waitKey(1)               
    def findObjContact(self):
        #return the object id of gripper contact object
        cpl=self.p.getContactPoints(self.blgripper,linkIndexA=0)
        cpr=self.p.getContactPoints(self.blgripper,linkIndexA=1)
        #TODO -  check contact forces also
        if(len(cpl)>0):
            return(cpl[0][2])
        elif(len(cpr)>0):
            return(cpr[0][2])    
        else:
            return(0)

    def detectObject(self,objId):
        #TODO: use a CNN to detect the object grasped from GelSight Data
        # as a temporary solution, a prior knowledge from pybullet env is used
        if objId in self.sphereIds:
            return("SPHERE")
        elif objId in self.legoIds:
            return("LEGO") 

    def checkTrayContact(self):
        #check whether the gripper makes contact with tray
        cpl=self.p.getContactPoints(self.blgripper,self.tray,0)
        cpr=self.p.getContactPoints(self.blgripper,self.tray,1)
        #TODO -  check contact forces also
        if((len(cpl)>0)or(len(cpr)>0)):
            #print("contact", cpl, cpr)
            return True
        else:
            return False

    def checkObjContact(self):
        #check whether the gripper makes contact with other objects
        cpl=self.p.getContactPoints(self.blgripper,linkIndexA=0)
        cpr=self.p.getContactPoints(self.blgripper,linkIndexA=1)
        #TODO -  check contact forces also
        if((len(cpl)>0)or(len(cpr)>0)):
            #print("contact", cpl, cpr)
            return True
        else:
            return False          

    def getGripForce(self):
        #get the gripper force
        joint_states=self.p.getJointStates(self.blgripper,np.arange(self.num_joints_g))
        f0=abs(joint_states[0][2][2])
        f1=abs(joint_states[1][2][2])
        force=np.round([f0,f1],3)
        return(force)

    def isObjInHand(self):
        #checkTrayContact()
        #returns True if the object is there inside the gripper
        #TODO -  check the distance between fingers
        #TODO - check whether both fingers are in contact with same body
        #TODO -  check for collision with other objects during grip, this can also increase Force
        force=self.getGripForce()
        f0=force[0]#abs(joint_states[0][2][2])
        f1=force[1]
        if((f0>self.GRASP_FTHRES)and (f1>self.GRASP_FTHRES)and (self.GraspCommanded==True)):
        #if(((f0+f1)>GRASP_FTHRES) and (GraspCommanded==True)):
            if((self.checkTrayContact()==True) and(self.checkObjContact==True) and (self.GraspCommanded==False)):
                return False
            else:
                #print("GRASPED")
                return True
        else:
            return False    

    def isGrasped(self):
        #checkTrayContact()
        #returns True if the object is there inside the gripper
        #TODO -  check the distance between fingers
        #TODO - check whether both fingers are in contact with same body
        #TODO -  check for collision with other objects during grip, this can also increase Force
        force=self.getGripForce()
        f0=force[0]#abs(joint_states[0][2][2])
        f1=force[1]
        if((f0>self.GRASP_FTHRES)and (f1>self.GRASP_FTHRES)and (self.GraspCommanded==True) and(self.prevGripCmd==True)):
        #if(((f0+f1)>GRASP_FTHRES) and (GraspCommanded==True)):
            if((self.checkTrayContact()==True) and(self.checkObjContact==True) and (self.GraspCommanded==False)):
                return False
            else:
                #print("GRASPED")
                return True
        else:
            return False   


    def closeGripper(self):
        self.GraspFlag = False
        self.GraspCommanded=True
        f_thres=self.GRASP_FTHRES
        f0=0
        f1=0
        count=0  
        count_thres=40  
        while((f0<f_thres) or (f1<f_thres)):
            #joint_states=p.getJointStates(blgripper,np.arange(num_joints_g))
            self.p.setJointMotorControlArray(self.blgripper, np.arange(self.num_joints_g), controlMode = self.p.POSITION_CONTROL, targetPositions = [0.5,-0.5], )
            self.p.stepSimulation()
            time.sleep(1.0/self.FREQ)
            #estimate the force on each finger
            force=self.getGripForce()
            f0=force[0]#abs(joint_states[0][2][2])
            f1=force[1]#abs(joint_states[1][2][2])
            count =count+1
            if count>=count_thres:
                break
            #print("GRC",count,f0,f1)
            self.drawline()
        if count>=count_thres:
            #print("NOGRIP")
            return 0
        else:
            #print("GRIP")
            self.GraspFlag =True
            return 1

    def openGripper(self):
        self.GraspCommanded = False
        self.GraspFlag = False    
        p_thres=0.005
        pos0=10
        pos1=10
        count_thres=40
        count=0
        while((pos0>p_thres) or(pos1>p_thres)):
            joint_states=self.p.getJointStates(self.blgripper,np.arange(self.num_joints_g))
            self.p.setJointMotorControlArray(self.blgripper, np.arange(self.num_joints_g), controlMode = self.p.POSITION_CONTROL, targetPositions = [0,0], )
            self.p.stepSimulation()
            time.sleep(1.0/self.FREQ)
            pos0=abs(joint_states[0][0])
            pos1=abs(joint_states[1][0])
            count =count+1
            if count>=count_thres:
                break
            #print(count, joint_states[0][0], joint_states[1][0])
            self.drawline()
        if count>=count_thres:
            return 0
        else:
            return 1
    def movetoPos(self,goal_pos):
        p_thres=0.001 #1 MM within commanded position
        error=10
        count=0
        count_thres=120 #timeout
        #while((px>p_thres)or(py>p_thres) or(pz>_p_thres)):
        while(error>p_thres):
            joint_pos = self.p.calculateInverseKinematics(self.kuka, self.num_joints_k-1, goal_pos, lowerLimits = self.low_limits, upperLimits = self.up_limits)
            self.p.setJointMotorControlArray(self.kuka, np.arange(self.num_joints_k), controlMode = self.p.POSITION_CONTROL, targetPositions = joint_pos, )

            self.p.stepSimulation()
            time.sleep(1.0/self.FREQ)
            link_state = self.p.getLinkState(self.kuka, self.num_joints_k-1,computeForwardKinematics=True)
            error = np.linalg.norm(goal_pos - np.array(link_state[4]))
            count=count+1
            if count>=count_thres:
                break
            #print(count, error)
            #print(count,link_state[4])
            self.drawline()
        
        if count>=count_thres:
            return 0
        else:
            return 1


    def getPose(self):
        #get the current end effector pose and orientation
        link_state = self.p.getLinkState(self.kuka, self.num_joints_k-1,computeForwardKinematics=True)
        return([link_state[4],link_state[5]])


    def getForces(self):
        #get the xyz forces on end effector
        forces=self.p.getJointState(self.kuka,self.num_joints_k-1)
        return([forces[2][0],forces[2][1],forces[2][2]])


    def movetoPosPose(self,goal_pos, goal_orn):
        p_thres=0.001 #1 MM within commanded position #TODO -  make it global
        o_thres = 0.1
        error=10
        error2 =10
        count=0
        count_thres=120 #timeout
        #while((px>p_thres)or(py>p_thres) or(pz>_p_thres)):
        while((error>p_thres) or(error2>o_thres)):
            joint_pos = self.p.calculateInverseKinematics(self.kuka, self.num_joints_k-1, targetPosition=goal_pos, targetOrientation=goal_orn, lowerLimits = self.low_limits, upperLimits = self.up_limits)
            self.p.setJointMotorControlArray(self.kuka, np.arange(self.num_joints_k), controlMode = self.p.POSITION_CONTROL, targetPositions = joint_pos, )

            self.p.stepSimulation()
            time.sleep(1.0/self.FREQ)
            link_state = self.p.getLinkState(self.kuka, self.num_joints_k-1,computeForwardKinematics=True)
            error = np.linalg.norm(goal_pos - np.array(link_state[4])) 
            error2= np.linalg.norm(goal_orn-np.array(link_state[5]))
            #print(count, error, error2)
            count=count+1
            if count>=count_thres:
                break
            #print(count, error)
            #print(count,link_state[4])
            self.drawline()
            #print("FORCE - ",getForces())
        #print("MOVPOS CN", count)
        if count>=count_thres:
            return 0
        else:
            return 1

    def moveStartPos(self):
            self.movetoPosPose([self.startX,self.startY,self.startZ],self.ORN_DOWN)

    def dumpObj(self,objectName):
        curPose=self.getPose()[0]
        if(objectName=="LEGO"):
            numSteps=2#move in steps to slow down #TODO -  make it global
            nwpos=list(self.getEquidistantPoints((curPose[0],curPose[1],self.ExtractZ), (self.LEGO_TRAY_X,self.LEGO_TRAY_Y,0.3), numSteps))
            #zp=np.arange(curPose[2],ExtractZ,(ExtractZ-curPose[2])/numSteps)
            for pos in nwpos:
                self.movetoPosPose([pos[0],pos[1],pos[2]],self.ORN_DOWN)
            #movetoPosPose([LEGO_TRAY_X,LEGO_TRAY_Y,0.3 ],orn_down)
        elif(objectName=="SPHERE"):
            numSteps=2 #move in steps to slow down
            nwpos=list(self.getEquidistantPoints((curPose[0],curPose[1],self.ExtractZ), (self.SPHERE_TRAY_X,self.SPHERE_TRAY_Y,0.3), numSteps))
            #zp=np.arange(curPose[2],ExtractZ,(ExtractZ-curPose[2])/numSteps)
            for pos in nwpos:
                self.movetoPosPose([pos[0],pos[1],pos[2]],self.ORN_DOWN)
        self.openGripper()     


    def updateObjMap(self):
        px=[]
        py=[]
        px2=[]
        py2=[]
        sval=[]
        s2val=[]
        s1val=[]
        #get proximity data
        proxDataRaw=self.getProximityData()
        curPose=self.getPose()[0]

        #TODO- make thrsholds global
        THR1=0.235
        THR2=THR1+self.PROX_SENS_RANGE_TIP
        sval=proxDataRaw[:len(self.SENS_ANGLES)]
        for i in range(self.SENS_LAYERS*len(self.SENS_ANGLES)):
            spos=self.p.getBasePositionAndOrientation(self.SENS_ID[i])#ray start pos
            epos=self.p.multiplyTransforms(spos[0],spos[1],(0,0,self.PROX_SENS_RANGE),[0,0,0,1]) #ray end pos
                
            test1=((self.TRAY_X+(self.TRAY_LEN/2.0))>=spos[0][0]>=(self.TRAY_X-(self.TRAY_LEN/2.0)))
            test2=((self.TRAY_Y+(self.TRAY_LEN/2.0))>=spos[0][1]>=(self.TRAY_Y-(self.TRAY_LEN/2.0)))
            test3=((self.TRAY_X+(self.TRAY_LEN/2.0))>=epos[0][0]>=(self.TRAY_X-(self.TRAY_LEN/2.0)))
            test4=((self.TRAY_Y+(self.TRAY_LEN/2.0))>=epos[0][1]>=(self.TRAY_Y-(self.TRAY_LEN/2.0)))
            if (test1 and test2 and test3 and test4):
                #append start pos
                px.append(spos[0][0])
                py.append(spos[0][1])
            
                #rayEndPos.append(epos[0])
                px.append(epos[0][0])
                py.append(epos[0][1])
                s1val.append(sval[i])
                #sval[i]=0
        i_prev=i
        #if above an z threshold,add button(tip) sensors too
        # this is to avoid false positives at exploration height
        if(THR2>=curPose[2]>=THR1):
            #sval=sval + proxDataRaw[-8:]
            sval=proxDataRaw[-8:]
            px=[]
            py=[]
            s1val=[]
            i_count=1
            #for i in range((len(proxDataRaw)-len(proxDataRaw[-8:])),len(proxDataRaw)):
            for i in range((len(self.SENS_POS)-8),len(self.SENS_POS)):    
                spos=self.p.getBasePositionAndOrientation(self.SENS_ID[i])
                epos=self.p.multiplyTransforms(spos[0],spos[1],(0,0,self.PROX_SENS_RANGE_TIP),[0,0,0,1]) #ray end pos
          
                test1=((self.TRAY_X+(self.TRAY_LEN/2.0))>=spos[0][0]>=(self.TRAY_X-(self.TRAY_LEN/2.0)))
                test2=((self.TRAY_Y+(self.TRAY_LEN/2.0))>=spos[0][1]>=(self.TRAY_Y-(self.TRAY_LEN/2.0)))
                test3=((self.TRAY_X+(self.TRAY_LEN/2.0))>=epos[0][0]>=(self.TRAY_X-(self.TRAY_LEN/2.0)))
                test4=((self.TRAY_Y+(self.TRAY_LEN/2.0))>=epos[0][1]>=(self.TRAY_Y-(self.TRAY_LEN/2.0)))
                #print("TIP",i)
                if (test1 and test2 and test3 and test4):
                    px.append(spos[0][0])
                    py.append(spos[0][1])
            
                    #rayEndPos.append(epos[0])
                    px.append(epos[0][0])
                    py.append(epos[0][1])
                    s1val.append(proxDataRaw[i_prev+i_count])
                    #print(i_prev+i_count)
                    i_count=i_count+1
            
        #double the array- this is because the px and py arrays contain both start and end pos
        for i in s1val:
            s2val.append(i)
            s2val.append(i)
        binsx=np.linspace((self.TRAY_X-(self.TRAY_LEN/2.0)),(self.TRAY_X+(self.TRAY_LEN/2.0)),self.MAP_ROWS+1)
        binsy=np.linspace((self.TRAY_Y-(self.TRAY_LEN/2.0)),(self.TRAY_Y+(self.TRAY_LEN/2.0)),self.MAP_COLS+1)

        dig_x=np.digitize(px, binsx)
        dig_y=np.digitize(py, binsy)
        self.ObjMap[dig_x-1,dig_y-1]=s2val


    def updateMap(self):

        
        #dummy readings for testing
        #px=np.random.uniform((TRAY_X-(TRAY_LEN/2.0)),(TRAY_X+(TRAY_LEN/2.0)), 5) # 5 nos of dummy readings
        #py=np.random.uniform((TRAY_Y-(TRAY_LEN/2.0)),(TRAY_Y+(TRAY_LEN/2.0)), 5) # 5 nos of dummy readings
        #get real readings from the start and end position of the proximity sensors
        rayStartPos=[]
        rayEndPos=[]
        px=[] # x coordinate of visited places (from position of ray start and ray end)
        py=[] # y coordinate of visited places
        for i in range(len(self.SENS_POS)):
            spos=self.p.getBasePositionAndOrientation(self.SENS_ID[i]) #ray start pos
            epos=self.p.multiplyTransforms(spos[0],spos[1],(0,0,self.PROX_SENS_RANGE),[0,0,0,1]) #ray end pos
            #check whether the spos and epos lies within limits
            test1=((self.TRAY_X+(self.TRAY_LEN/2.0))>=spos[0][0]>=(self.TRAY_X-(self.TRAY_LEN/2.0)))
            test2=((self.TRAY_Y+(self.TRAY_LEN/2.0))>=spos[0][1]>=(self.TRAY_Y-(self.TRAY_LEN/2.0)))
            test3=((self.TRAY_X+(self.TRAY_LEN/2.0))>=epos[0][0]>=(self.TRAY_X-(self.TRAY_LEN/2.0)))
            test4=((self.TRAY_Y+(self.TRAY_LEN/2.0))>=epos[0][1]>=(self.TRAY_Y-(self.TRAY_LEN/2.0)))
            if( test1 and test2 and test3 and test4):
            #rayStartPos.append(spos[0])
                px.append(spos[0][0])
                py.append(spos[0][1])
            
                #rayEndPos.append(epos[0])
                px.append(epos[0][0])
                py.append(epos[0][1])
            else:
                pass    
            
        binsx=np.linspace((self.TRAY_X-(self.TRAY_LEN/2.0)),(self.TRAY_X+(self.TRAY_LEN/2.0)),self.MAP_ROWS+1)
        binsy=np.linspace((self.TRAY_Y-(self.TRAY_LEN/2.0)),(self.TRAY_Y+(self.TRAY_LEN/2.0)),self.MAP_COLS+1)

        dig_x=np.digitize(px, binsx)
        dig_y=np.digitize(py, binsy)
        self.VisitMap[dig_x-1,dig_y-1]=1


    def updateCurPosMap(self):
        self.CurPosMap =  np.zeros((self.MAP_ROWS,self.MAP_COLS),dtype = np.uint8) #to make everything zero and flush previous data
        curPose=self.getPose()[0]
        #check whether curPos is inside workspace
        test1=((self.TRAY_X+(self.TRAY_LEN/2.0))>=curPose[0]>=(self.TRAY_X-(self.TRAY_LEN/2.0)))
        test2=((self.TRAY_Y+(self.TRAY_LEN/2.0))>=curPose[1]>=(self.TRAY_Y-(self.TRAY_LEN/2.0)))
        if(test1 and test2):
            binsx=np.linspace((self.TRAY_X-(self.TRAY_LEN/2.0)),(self.TRAY_X+(self.TRAY_LEN/2.0)),self.MAP_ROWS+1)
            binsy=np.linspace((self.TRAY_Y-(self.TRAY_LEN/2.0)),(self.TRAY_Y+(self.TRAY_LEN/2.0)),self.MAP_COLS+1)
            dig_x=np.digitize(curPose[0], binsx)
            dig_y=np.digitize(curPose[1], binsy)
            self.CurPosMap[dig_x-1,dig_y-1]=1


    def calculateRewards(self):
        DoneFlag=False
        r1=0
        r2=0
        r3=0
        r4=0
        r5=0
        p1=0
        p2=0
        p3=0
        p4=0
        p5=0
        p6=0
        p7=0
        
        

        #-----------  REWARD TO MOVE DOWN TOWARDS TRAY DURING START ---------------##
        # If no object grasped, then  -1 x |gripper_height-tray_height| . 
        # ( this is to learn the robot to move down, towards the tray during initialization )  
        # TODO - once reached traypos-> set flag; if go down anytime-> penalty; if go up, 0 reward
        #TODO - if not grasped and above a particular height - penalize
        curPose=self.getPose()[0]
        self.GraspFlag =self.isGrasped()
        self.objInHandFlag=self.isObjInHand()

        if((self.GraspFlag==False)and(self.reachedTrayFlag==False)):
            downscore=(curPose[2]-self.GripperTrayZ)
            if ((abs(downscore)<=self.deltaPoseZSmall) ):
                self.reachedTrayFlag =True
                r1=self.ReachedZTrayReward
            elif ((self.prevZ-curPose[2])<self.deltaPrevZ):
                r1=self.StepPenalty
            else:
                r1=self.DownMoveReward  

        if((self.GraspFlag==False)and(self.reachedTrayFlag==True)):
            downscore=(curPose[2]-self.GripperTrayZ)
            if ((abs(downscore)>=(2*self.deltaPoseZSmall)) ):
                r1=self.StepPenalty
        '''
        if((self.GraspFlag==False)):# and(reachedTrayFlag==False)):
            downscore=(curPose[2]-self.GripperTrayZ)
            if ((abs(downscore)<=self.deltaPoseZSmall) and (self.reachedTrayFlag==False)):
                self.reachedTrayFlag =True
                r1=self.ReachedZTrayReward
            
            elif(self.reachedTrayFlag==True):
                if(downscore< (-self.deltaPoseZ)): #if going lower- penalize
                    r1=-self.scaleRZ*abs(downscore)
                else:
                    #r1=0 #
                    if(downscore>self.deltaPoseZHigh):
                        r1= -self.scaleRZ*abs(downscore) #if go up after reaching tray pose, higher than deltaPoseZHigh
                    elif(downscore>0):
                        r1=-self.scaleRZLow*abs(downscore)#0  #if go up after reaching tray pose-( but still below deltaPoseZHigh) no reward ( this is not to penalize learning of moveing up for grasping)
            elif(self.reachedTrayFlag==False):
                if((self.prevZ-curPose[2])<self.deltaPrevZ):
                    r1=self.StepPenalty
                else:
                    r1=5#-self.scaleRZ*abs(downscore) # initial moving down incentive 
        '''            
        self.prevZ=curPose[2]            

        #-------------- REWARD TO MOVE TO TOP EXTRACTION POINT AFTER GRASPING ------------##    
        #High Positive if object grasped and -1 x |gripper_height - goal_height|  
        #  A constant + above reward. ie, contant as reward for closing the gripper and above reward for moving up 
        #TODO ? should we get rid of this and hardcode dumping once grasped
        #TODO- check for prevGripCmd when learning to move up
        if((self.GraspFlag==True) ):
            '''
            #3 cases
            #case 1 - gripped -  constant reward #given only once, else agent may learn to cheat by moving sideways for high rewards instead of up
            if((self.previousGraspFlag==False) and (LEARN_MOVE_UP==True)): ##########<<== THIS IS DEFUNCT<<<<<<<<<<<<<<<<<<<<<
                r2=GripReward
                previousGraspFlag  = True
                checkDropped = True
            elif(LEARN_MOVE_UP==True):##########<<== THIS IS DEFUNCT <<<<<<<<<<<<<<<<<<
                upscore= (abs(curPose[2]-ExtractZ))
                if (upscore<=deltaPoseZ):
                    #case 3 - gripped, moved up to position=> reset previousGraspFlag
                    r2=ReachZUpReward 
                    #openGripper()#optional
                    objId = findObjContact()
                    if(objId>0):                  
                        #print("DETECTED",detectObject(objId))
                        dumpObj(detectObject(objId))
                        moveStartPos()
                    if(areaCovered>=99): #DONE
                    
                        r4=DoneReward
                        
                        
                    previousGraspFlag = False
                    reachedTrayFlag  =False
                    #  -run the CNN to classify object, sort it to respective tray
                    #           after that reset the ee position to startZ pos
                    #       -check of area covered = 100% => FINAL REWARD and DONE
                    
                else:
                    #case 2 - gripped and still have to move up; reward becomes more less negative as we move up
                    r2= -scaleRZ * upscore

            elif((LEARN_MOVE_UP==False) and (prevGripCmd==True)):
            '''   
            if(self.prevGripCmd==True):
                
                #move up to extract position
                numSteps=3 #move in steps to slow down #TODO - define in class
                nwpos=list(self.getEquidistantPoints((curPose[0],curPose[1],curPose[2]), (curPose[0],curPose[1],self.ExtractZ), numSteps))
                #zp=np.arange(curPose[2],ExtractZ,(ExtractZ-curPose[2])/numSteps)
                for pos in nwpos:
                    self.movetoPosPose([pos[0],pos[1],pos[2]],self.ORN_DOWN)
                #detect object
                time.sleep(0.15)
                objId = self.findObjContact()
                if(objId>0):
                    r2=self.GripReward
                    self.dumpObj(self.detectObject(objId))
                    self.GraspCount=self.GraspCount+1
                    #r2=2*r2
                    self.moveStartPos()                   
                self.reachedTrayFlag  =False

        if(self.GraspCount==self.totalObjs): #DONE #check whether all objects have been picked up
            #TODO return DONE or make DONE flag True and reset
            r4=self.DoneReward
            self.DoneFlag=True    


        #-------------- PENALTY FOR DROPPING ------------------------------------------#
        if((self.objInHandFlag==False) and (self.previousGraspFlag==True) and (self.checkDropped==True)): #ALMOST DEFUNCT NOW   
            p1=-20
            self.checkDropped=False
            self.reachedTrayFlag =False
            self.previousGraspFlag  = False
            

        #-------------- REWARD TO EXPLORE MORE AREA INSIDE TRAY --------------------------    
        # check whether the gripper is in bottom position as well as has not grasped anything
        if(((self.GripperTrayZ-self.deltaPoseZ)<=curPose[2]<=(self.GripperTrayZ+self.deltaPoseZ)) and (self.GraspFlag ==False)and (self.reachedTrayFlag ==True)):
            self.updateMap()
            self.areaCovered=(np.sum(self.VisitMap)/(self.MAP_COLS*self.MAP_ROWS)) * 100 #percentage of area covered
            deltaMapArea=(self.areaCovered-self.prevMapArea)
            self.prevMapArea = self.areaCovered
            r3=deltaMapArea * self.MapDeltaReward#  TODO , scale factor, should we give area coverd also as observation input?
            if(self.areaCovered>=99.0):
                #TODO -
                r5=self.MapCoverReward
                #reset map for next pass
                self.VisitMap = np.zeros((self.MAP_ROWS,self.MAP_COLS),dtype = np.uint8) #clear the map. so that the agent may be incentivized to explore more 

        
        #-------------- PENALTY FOR COLLISION WITH OTHER OBJECTS & TRAY ------------------
        if(self.checkTrayContact()==True):
            p2=self.TrayContactPenalty
        if((self.checkObjContact()==True) and (self.GraspFlag==False)):
            # - this method still has a bug.. if gripper grasps an objects and moves around making 
            #       collisions, it wont be penalized
            p3=self.ObjContactPenalty
        #--------------- PENALTY FOR WANDERING OFF THE XY LIMITS OF TRAY-----------------
        # TRAY CONTACT TEST 
        #y axis contact at 0.185 & -0.186
        #x axis- gripper open -  0.464 & 0.736
        #x axis- gripper closed 0.417 &0.786
        # z axis reset if below 0.175
        if(curPose[0]<self.X_MIN):
            p4=-self.scaleRXY*(abs(curPose[0]-self.X_MIN))  
        elif(curPose[0]>self.X_MAX):
            p4=-self.scaleRXY*(abs(curPose[0]-self.X_MAX))
        if(curPose[1]<self.Y_MIN):
            p5=-self.scaleRXY*(abs(curPose[1]-self.Y_MIN))  
        elif(curPose[1]>self.Y_MAX):
            p5=-self.scaleRXY*(abs(curPose[1]-self.Y_MAX))
        if(curPose[2]<self.Z_MIN):
            p7= -1000
            self.DoneFlag=True 
            #self.reset()
        if(curPose[2]>self.Z_MAX):
            p7=-self.scaleRXY*(abs(curPose[2]-self.Y_MAX))
        #-------- RESET if WANDERING TOO MUCH
        #
        if(curPose[0]<(self.X_MIN-0.01)):
            p4=-1000
            self.DoneFlag=True
        if(curPose[0]>(self.X_MAX+0.01)):
            p4=-1000
            self.DoneFlag=True    
        if(curPose[1]<(self.Y_MIN-0.01)): 
            p5=-1000
            self.DoneFlag=True     
        if(curPose[1]>(self.Y_MAX+0.01)): 
            p5=-1000
            self.DoneFlag=True
        if(curPose[2]>(self.Z_MAX+0.006)):
            p7= -1000
            self.DoneFlag=True 


        #------------- PENALTY FOR COLLISION FROM TOP DIRECTION (by measuring FT sensor)
        forces=np.array(self.getForces())
        if(forces[2]>self.ZFthres):
            p6=self.ZContactPenalty

        #-------------- REWARD FOR MOVING TO UNEXPLORED CELLS IN THE MAP---------------
        # TODO - Do we really need it??    

        rew=r1+r2+r3+r4+r5+p1+p2+p3+p4+p5+p6+p7
        #print(rew, end='\r')
        #print("Reward","{:.3f}".format(rew),"{:.3f}".format(forces[0]),"{:.3f}".format(forces[1]),"{:.3f}".format(forces[2]),"Pos " ,"{:.3f}".format(curPose[0]),"{:.3f}".format(curPose[1]),"{:.3f}".format(curPose[2]),end='\r')
        
        #print(self.StepCount, rew," MovDwn","{:.3f}".format(r1),"GripRwd","{:.3f}".format(r2),p1,"CONT: ",p2+p3,"MAPCVR: ", "{:.3f}".format(r3),self.areaCovered,r5, "WNDR: ","{:.3f}".format(p4+p5),"ZCONT",p6,"{:.3f}".format(curPose[2])," DNE ",self.DoneFlag, end='\r')
        return(rew) 


    def getObservation(self):
        obs=[]
        #global visitMap
        #get Proximity Sensor Data
        proxDataRaw=self.getProximityData()
        proxData=[]
        for i in range((self.SENS_LAYERS*len(self.SENS_ANGLES))):
            proxData.append(proxDataRaw[i])
        #append the tip sensors    
        proxData.append(proxDataRaw[-8])
        proxData.append(proxDataRaw[-7])
        proxData.append(proxDataRaw[-6])
        proxData.append(proxDataRaw[-5])
        proxData.append(proxDataRaw[-4])
        proxData.append(proxDataRaw[-3])
        proxData.append(proxDataRaw[-2])
        proxData.append(proxDataRaw[-1])

        proxData=np.array(proxData, dtype = np.uint8)
        obs.append(proxData)

        

        #current pos
        curPose=np.array(self.getPose()[0],dtype = np.float)
        if (self.StepCount ==0):
            self.posStack = np.stack((curPose,curPose, curPose)).reshape(9,)
        else:
            self.posStack=np.append(curPose,self.posStack[:6]).reshape(9,)    
        #curPose=np.array(self.getPose()[0],dtype = np.float)
        obs.append(self.posStack)

        #get the wrist Force sensor data
        forces=self.getForces()
        wristforce=np.array(forces,dtype = np.float)
        obs.append(wristforce)

        #return ObjMap
        self.updateObjMap()
        #return quantized xy position
        self.updateCurPosMap()
        obs.append(self.ObjMap)#+self.CurPosMap) 
        #get Map of visited areas
        obs.append(self.VisitMap)#+self.CurPosMap)
        obs.append(self.CurPosMap)
        
        # obs.append(self.CurPosMap)

        #get gelsight data
        im1,im2=self.getGelSightData()
        im1=im1[0:240,80:320]
        im1=cv2.resize(im1,(32,32),interpolation = cv2.INTER_AREA)
        im1=np.array(im1 )#* 255, dtype = np.uint8)
        im2=im2[0:240,80:320]
        im2=cv2.resize(im2,(32,32),interpolation = cv2.INTER_AREA)
        im2=np.array(im2 )#* 255, dtype = np.uint8)
        obs.append(im1)
        obs.append(im2)
        #[proxData,curPos[3steps],wristForce,ObjMap, VisitMap,CurPosMap,GelSight1_im, GelSight2_im]
        return(obs)



    def step(self,action):
        self.prevGripCmd=False
        self.StepCount=self.StepCount+1
        
            
        delta_x=action[0]
        delta_y=action[1]
        delta_z=action[2]
        gripper_cmd=action[3]

        #get current EE position
        link_state = self.getPose()[0] #p.getLinkState(kuka, num_joints_k-1,computeForwardKinematics=True)
        cur_pos=np.array(link_state)
        #newpos=current EE pos + delta
        new_pos=np.array([delta_x,delta_y,delta_z])+cur_pos
        #move robot to new pos
        self.movetoPosPose(new_pos,self.ORN_DOWN)
        # move gripper
        if(gripper_cmd==-1):
            self.openGripper()
        elif(gripper_cmd==1):
            self.closeGripper()
            self.prevGripCmd=True
        #calculate rewards
        rew=self.calculateRewards()
        #sense the state (observe)
        obs=self.getObservation()
        # return reward, isDone, SensorObservations(next state)
        if(self.StepCount==self.MaxSteps):
            self.StepCount=0
            #end of episode
            self.DoneFlag = True
        return(rew,self.DoneFlag,obs)


    def reset(self):
        #remove all objects, reset everything
        self.removeAll()
        #re-init
        self.p.setRealTimeSimulation(0)
        self.FREQ=240.0
        self.p.setGravity(0,0,-10.0)
        #load robot
        self.loadRobot()
        #reinitialize random position of objects
        self.loadObjs()
        #move the robot arm to startpos
        self.moveStartPos()
        
        #TODO clear all flags - doneflag,
        global previousGraspFlag
        global reachedTrayFlag
        global GraspFlag
        global checkDropped
        global VisitMap  
        global prevMapArea
        global ObjMap
        global areaCovered
        global objInHandFlag
        global GraspCount
        global StepCount
        global prevZ
        global DoneFlag
        global prevGripCmd
        global GraspCommanded

        self.previousGraspFlag = False
        self.reachedTrayFlag=False
        self.GraspFlag = False
        self.checkDropped = True
        self.VisitMap = np.zeros((self.MAP_ROWS,self.MAP_COLS),dtype = np.uint8) #for every visited tile, the corresponding matrix entry is marked 1 
        self.ObjMap=np.zeros((self.MAP_ROWS,self.MAP_COLS),dtype = np.uint8)
        self.prevMapArea=0
        self.areaCovered=0
        self.objInHandFlag=False
        self.GraspCount=0
        self.StepCount =0
        
        self.prevZ=self.startZ
        self.DoneFlag=False
        self.prevGripCmd=False
        self.GraspCommanded = False

        #return the state observation
        return(self.getObservation())  
        
    def getEquidistantPoints(self,p1, p2, numpoints):
        return zip(np.linspace(p1[0], p2[0], numpoints+1),
                np.linspace(p1[1], p2[1], numpoints+1),
                np.linspace(p1[2], p2[2], numpoints+1))
    
    def disconnect(self):
        self.p.disconnect()

    def clamp(self,n, minn, maxn):
        return max(min(maxn, n), minn)    

    
