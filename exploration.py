#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose2D
import numpy as np
from asl_turtlebot.msg import DetectedObject
import tf
import json

class exploration:
    def __init__(self):
        self.counter = 0
        self.published = False
        self.first = False
        self.explored = False
        self.goal = Pose2D()
        self.dictionary = {
            "fire_hydrant": np.array([0,0]).reshape(1,2),
            "potted_plant": np.array([0,0]).reshape(1,2),
            "stop_sign": np.array([0,0]).reshape(1,2),
            "street_sign": np.array([0,0]).reshape(1,2),
            # "airplane": np.array([0,0]).reshape(1,2),
        }
        ## Following should be used:
        # self.goal_list_x = [3.35, 2.5, 0.8, 0.35, 0.34, 0.35, 1.3, 2.35, 2.35, 1.5, 2.32, 2.32, 3.45, 3.45, 3.15, 3.15]
        # self.goal_list_y = [2.80, 2.8, 2.8, 2.20, 1.60, 0.40, 0.4, 0.40, 1.60, 1.6, 1.60, 0.40, 0.40, 1.0, 1.0, 1.6]
        # self.goal_list_th = [3/4*np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2, 0, 0, np.pi/2, -np.pi, 0, 0, 0, 0, np.pi/2, np.pi/2, 0]

        self.goal_list_x = [3.35, 0.8, 0.35, 0.35, 2.35, 1.5, 2.32, 3.45, 3.15]
        self.goal_list_y = [2.80, 2.8, 2.20, 0.40, 0.40, 1.6, 0.40, 1.0, 1.6]
        self.goal_list_th = [3/4*np.pi, -np.pi, -np.pi/2, 0, np.pi/2, 0, 0, np.pi/2, 0]

        
        
        # self.fire_hydrant = []
        # self.tree = []
        # self.__ = []
        # self.__ = []
        # self.__ = []
        self.x = 0
        self.y = 0
        self.theta = 0

        rospy.init_node("Explore",disable_signals=True)
        self.trans_listener = tf.TransformListener()
        self.pub = rospy.Publisher('/cmd_nav',Pose2D,queue_size=1)
        self.sub = rospy.Subscriber('/cmd_vel',Twist,self.cmd_vel_callback)
        self.sub_detect_fire = rospy.Subscriber('/detector/fire_hydrant',DetectedObject,self.detector_callback)
        self.sub_detect_tree = rospy.Subscriber('/detector/potted_plant',DetectedObject,self.detector_callback)
        self.sub_detect_stop = rospy.Subscriber('/detector/stop_sign',DetectedObject,self.detector_callback)
        # self.sub_detect_can = rospy.Subscriber('/detector/traffic_light',DetectedObject,self.detector_callback)
        self.sub_detect_quad = rospy.Subscriber('/detector/airplane',DetectedObject,self.detector_callback)
        self.pub_rescue = rospy.Publisher('/rescue',String,queue_size=1)
        self.pub_exploration = rospy.Publisher('/exploration',Bool,queue_size=1)
                
        

    def cmd_vel_callback(self,data):
        if self.counter >= len(self.goal_list_x):
            print('finished exploraton')
            (translation,rotation) = self.trans_listener.lookupTransform("/map","/base_footprint",rospy.Time(0))
            self.x = translation[0]
            self.y = translation[1]
            if np.linalg.norm(np.array([self.x, self.y])-np.array([3.15,1.6])) <= 0.05 and self.explored == False:
                print('calculate position')
                for key in self.dictionary:
                    xy = self.dictionary[key][1:,:]
                    xy_average = np.mean(xy,axis=0)
                    self.dictionary[key] = (xy_average[0],xy_average[1])
                self.explored = True
            if self.explored == True:
                print("now_publish")
                self.pub_exploration.publish(True)
                self.pub_rescue.publish(json.dumps(self.dictionary))
                # rospy.signal_shutdown('Done!')
        elif self.counter < len(self.goal_list_x):
            if data.linear.x == 0 and data.angular.z == 0 and self.published == False:
                self.goal = Pose2D()
                self.goal.x = self.goal_list_x[self.counter]
                self.goal.y = self.goal_list_y[self.counter]
                self.goal.theta = self.goal_list_th[self.counter]
                self.published = True
            if data.linear.x != 0 and self.published == True:
                self.counter += 1
                self.published = False
                print(self.counter)
            self.pub.publish(self.goal)


    def detector_callback(self,data):
        (translation,rotation) = self.trans_listener.lookupTransform("/map","/base_footprint",rospy.Time(0))
        self.x = translation[0]
        self.y = translation[1]
        euler = tf.transformations.euler_from_quaternion(rotation)
        self.theta = euler[2]
        
        obj_x,obj_y = self.compute_position(data)
        if self.counter == 2 and data.name == "fire_hydrant":
            if data.distance <= 0.7:
                print(data.name)
                self.dictionary[data.name] = np.append(self.dictionary[data.name],[[obj_x,obj_y]],axis=0)
                # print([obj_x,obj_y])
        elif self.counter == 4 and data.name == "stop_sign":
            if data.distance <= 0.7:
                print(data.name)
                if self.dictionary[data.name].shape[0] < 40: self.dictionary[data.name] = np.append(self.dictionary[data.name],[[obj_x,obj_y]],axis=0)
                # print([obj_x,obj_y])
        elif self.counter == 6 and data.name == "potted_plant":
            if data.distance <= 0.7:
                print(data.name)
                if self.dictionary[data.name].shape[0] < 30: self.dictionary[data.name] = np.append(self.dictionary[data.name],[[obj_x,obj_y]],axis=0)
                # print([obj_x,obj_y])
        elif self.counter == 7 and data.name == "stop_sign":
            if data.distance <= 0.7:
                data.name = "street_sign"
                print(data.name)
                if self.dictionary[data.name].shape[0] < 40: self.dictionary[data.name] = np.append(self.dictionary[data.name],[[obj_x,obj_y]],axis=0)
                # print([obj_x,obj_y])
        # elif (self.counter == 8 or self.counter == 9) and data.name == "airplane":
        #     if data.distance <= 1:
        #         print(data.name)
        #         self.dictionary[data.name] = np.append(self.dictionary[data.name],[[obj_x,obj_y]],axis=0)
                # print([obj_x,obj_y])


    def compute_position(self,data):
        d = data.distance
        theta_l = data.thetaleft
        theta_r = data.thetaright
        if theta_l > np.pi: theta_l -= 2*np.pi
        if theta_r > np.pi: theta_r -= 2*np.pi
        theta_m = 1/2*(theta_l+theta_r)
        gamma = self.theta+theta_m
        # if gamma < 0: gamma += 2*np.pi
        # if gamma > 2*np.pi: gamma -= 2*np.pi

        obj_x = self.x+d*np.cos(gamma)
        obj_y = self.y+d*np.sin(gamma)
        print([d,[obj_x,obj_y]],"\n")
        return obj_x,obj_y




    def run(self):
        rospy.spin()

if __name__ == '__main__':
    exp = exploration()
    exp.run()
    