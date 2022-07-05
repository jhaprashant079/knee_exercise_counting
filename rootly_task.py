import cv2
import os
libararies=["opencv-python","mediapipe", "vg", "pandas", "matplotlib", "sympy", "mpmath"]
for library in libararies:
    os.system(f"py -m pip install \"{library}\"")


import mediapipe as mp
from math import atan
import math
import vg
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import sympy
import mpmath

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("KneeBendVideo.mp4")

frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output.mp4',fourcc, fps, (frameWidth,frameHeight))

state ="straight"#straight/bent
start=0.0
end=0.0
timer="pause"
tmp=1#boolean to know has the leg state gone through straight after the last counting 
ssc = 0# slacking_straight_count , count of frames when leg is straight without completion of timer
hold_duration=8
rep_count=0
message=""
fluctuated_count=0
fluct_threshold=35
fluctuation_count_thr=4
last_unfluctuated = [] #0--> leg, 1 -->angle

n=3
beta=(n-1)/n #it's 0.8 for 5
moving_angle2d_average=0 

stat_df = pd.DataFrame(columns=["time", "angle"])


def calculateAngle(x1, y1, z1,
                   x2, y2, z2,
                   x3, y3, z3):
                        
    # Find direction ratio of line AB
    ABx = x1 - x2;
    ABy = y1 - y2;
    ABz = z1 - z2;
 
    # Find direction ratio of line BC
    BCx = x3 - x2;
    BCy = y3 - y2;
    BCz = z3 - z2;
    vec1=np.array([ABx,ABy,ABz])
    vec2=np.array([BCx,BCy,BCz])
    ans = vg.angle(vec1, vec2)
    return ans

count = 0
with mp_pose.Pose(static_image_mode=False,model_complexity=2,min_detection_confidence=0.5,min_tracking_confidence=0.5,smooth_landmarks = True) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty video frames.")
            continue
        
        count+=1
        print(count)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks == None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Test vdo', image)
            continue
        left=[results.pose_landmarks.landmark[23],results.pose_landmarks.landmark[25],results.pose_landmarks.landmark[27]] # [hip,knee,ankle]
        right=[results.pose_landmarks.landmark[24],results.pose_landmarks.landmark[26],results.pose_landmarks.landmark[28]] # [hip,knee,ankle]

        #figuring out which leg is closer to camera
        leg=[] #left/right
        if left[0].z < right[0].z:
            leg=left
        else:
            leg=right
        
        #finding angle at knee
        l1=sympy.Line((leg[0].x,leg[0].y),(leg[1].x,leg[1].y))
        l2=sympy.Line((leg[1].x,leg[1].y), (leg[2].x,leg[2].y))
        bent_angle2d= 180-math.floor(sympy.N(mpmath.degrees(l1.angle_between(l2))))
        
        bent_angle3d = int(calculateAngle(leg[0].x,leg[0].y,leg[0].z,
                                    leg[1].x,leg[1].y,leg[1].z,
                                    leg[2].x,leg[2].y,leg[2].z))
        
        
        if count==1:
            moving_angle2d_average=bent_angle2d
            last_unfluctuated = (leg,bent_angle3d)
        else:
            if abs(bent_angle3d-last_unfluctuated[1])>fluct_threshold and fluctuated_count <=fluctuation_count_thr:
                #no change in last_unfluctuated
                fluctuated_count+=1
            else:
                fluctuated_count=0
                last_unfluctuated = [leg,bent_angle3d]

        leg,bent_angle3d_mod=last_unfluctuated
        moving_angle2d_average = beta*moving_angle2d_average+(1-beta)*bent_angle2d

        bent_angle=bent_angle2d
        df_tmp = {"time": cap.get(cv2.CAP_PROP_POS_MSEC)/1000,"angle":bent_angle}
        stat_df = stat_df.append(df_tmp, ignore_index=True)

        angle_to_show = moving_angle2d_average if bent_angle3d_mod >=80 else bent_angle3d_mod
        #finding rep count
        message = ""
        if bent_angle <= 140 :
            state="bent"
            end=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            timer=int(end-start)
            if timer > hold_duration:
                if tmp == 1:
                    rep_count+=1
                    tmp=0
                    ssc=0
                #if last timer was pause then count ++ else may put message to straighten his leg
                message = "Timer complete"

        else:
            if timer!="pause" :
                if timer < hold_duration and ssc <= 15:
                    message="keep your knee bent"
                    ssc+=1

                else:
                    tmp=1
                    ssc=0
                    state = "straight"
                    start=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    timer="pause"
            else:
                start=cap.get(cv2.CAP_PROP_POS_MSEC)/1000

        #writing different things at image
        image.flags.writeable = True
        leg_pixel=[mp_drawing._normalized_to_pixel_coordinates(e.x, e.y, frameWidth, frameHeight) for e in leg]
        for landmark in leg_pixel:
            cv2.circle(image,landmark , 5, (0, 255, 0), -1)
        cv2.line(image, leg_pixel[0],leg_pixel[1],(255,0,0),1)
        cv2.line(image, leg_pixel[1],leg_pixel[2],(255,0,0),1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(image,"angle: "+str(int(angle_to_show)),(int(frameWidth*3/5),int(frameHeight/5)), font, 1,(204,0,102),2,cv2.LINE_AA)
        cv2.putText(image,"rep_count: "+str(rep_count),(int(frameWidth*3/5),int(frameHeight/5)-32*1), font, 1,(204,0,102),2,cv2.LINE_AA)        
        cv2.putText(image,"timer: "+str(timer),(int(frameWidth*3/5),int(frameHeight/5)-32*2), font, 1,(204,0,102),2,cv2.LINE_AA)        
        cv2.putText(image,message,(int(frameWidth/25),int(frameHeight*24/25)), font, 1,(0,255,0),2,cv2.LINE_AA)        

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)
#         cv2.imshow('Test vdo', image)
        if cv2.waitKey(1) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
stat_df.drop(stat_df[stat_df['time'] == 0].index, inplace = True)

plt.plot(stat_df["time"],stat_df["angle"])
plt.xlabel('time(in sec)')
plt.ylabel('bend angle(in degree)')
plt.title('bend stats')
plt.savefig("stat.png",bbox_inches ="tight")
image=cv2.resize(cv2.imread("stat.png"), (frameWidth,frameHeight))
for _ in range(150):
    out.write(image)

bend_percent=len(stat_df[stat_df["angle"]<140])/len(stat_df)*100
img = cv2.rectangle(image, (0,0), (frameWidth,frameHeight), (0,0,0), -1)
cv2.putText(img,"Knee bent ratio(to total time) : ",(int(frameWidth*1/10),int(frameHeight*3/5)), font, 1,(255,255,255),2,cv2.LINE_AA)
cv2.putText(img,str(int(bend_percent))+"%",(int(frameWidth*1/10),int(frameHeight*3/5)+32), font, 1,(255,255,255),2,cv2.LINE_AA)
for _ in range(100):
    out.write(img)

out.release()

#output video drive link(in case it fails to upload on form)
#https://drive.google.com/file/d/1L4xQp6x5BksfPK49zfV5ULmk8Ym94IbW/view?usp=sharing