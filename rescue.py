#!/usr/bin/env python3

import rospy
import numpy as np
from sys import maxsize
from itertools import permutations
from std_msgs.msg import Int16MultiArray, String
from geometry_msgs.msg import Twist, Pose2D
import json
import navigator
from navigator import Navigator
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from utils.grids import StochOccupancyGrid2D

class rescue:
    def __init__(self):
        rospy.init_node("Rescue",disable_signals=True)
        self.rescue_list = []
        self.dictionary = {}
        self.counter = 0
        self.index_dict = {
            1: "fire_hydrant",
            2: "stop_sign",
            3: "street_sign",
            4: "potted_plant",
            # 5: "airplane"
        }
        self.starting_position = (3.15,1.6,0)
        self.pos_array = [self.starting_position]
        self.goal_list = None
        self.published = False
        self.loaded = False
        self.know = False
        self.compute = False
        self.goal = Pose2D()

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 15
        self.num_samples = 500
        self.gamma = 4
        self.pts_per_line = 5

        self.v_des = 0.1
        self.spline_alpha = 0.1
        self.traj_dt = 0.1

        self.pub = rospy.Publisher('/cmd_nav',Pose2D,queue_size=1)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        self.sub_rescue_dict = rospy.Subscriber('/rescue',String,self.rescue_dict_callback)
        self.sub_rescue_list = rospy.Subscriber('/rescue_list',Int16MultiArray, self.rescue_list_callback)
        self.sub = rospy.Subscriber('/cmd_vel',Twist,self.cmd_vel_callback)



    def rescue_dict_callback(self,msg):
        if not self.loaded:
            self.dictionary = json.loads(msg.data)
            for key in self.dictionary:
                self.dictionary[key][0] += 0.1
                self.dictionary[key][1] += 0.05
                if self.dictionary[key][0] < 0.4: self.dictionary[key][0] = 0.4
                if self.dictionary[key][0] > 3.4: self.dictionary[key][0] = 3.4
                if self.dictionary[key][1] < 0.4: self.dictionary[key][1] = 0.4
                if self.dictionary[key][1] > 2.6: self.dictionary[key][1] = 2.6
                if key == "potted_plant": self.dictionary[key] = (2.3,1.6)
            self.loaded = True

        if self.know and self.loaded and (not self.compute) and (self.occupancy is not None):
            self.goal_list = self.tsp()
            self.compute = True

    def rescue_list_callback(self,msg):
        if not self.know:
            print(msg.data)

            self.rescue_list = msg.data
            self.know = True

    def cmd_vel_callback(self,data):
        if self.goal_list is not None:
            # if self.counter >= len(self.goal_list):
            #     rospy.signal_shutdown('Finish Rescue!')
            if self.counter < len(self.goal_list):
                if data.linear.x == 0 and data.angular.z == 0 and self.published == False:
                    print("time to rescue")
                    rospy.sleep(2)
                    self.goal.x = self.goal_list[self.counter][0]
                    self.goal.y = self.goal_list[self.counter][1]
                    self.goal.theta = self.goal_list[self.counter][2]
                    self.published = True
                if data.linear.x != 0 and self.published == True:
                    self.counter += 1
                    self.published = False
                    print(self.counter)
                self.pub.publish(self.goal)

    def tsp(self):
        self.pos_array = [self.starting_position]
        angle_array = [0,4*np.pi/5,-3*np.pi/4,-3*np.pi/4,3*np.pi/4]
        for objects in self.rescue_list:
            position = self.dictionary[self.index_dict[objects]]
            self.pos_array.append((position[0],position[1],angle_array[objects]))
        # print(self.pos_array)
        # obj_1, obj_2, obj_3 = self.rescue_list
        # obj_1, obj_2, obj_3 = self.index_dict[obj_1], self.index_dict[obj_2], self.index_dict[obj_3]
        # self.pos_array = [self.starting_position, self.dictionay[obj_1], self.dictionay[obj_2], self.dictionay[obj_3]]
        length = len(self.rescue_list)
        self.time_matrix = np.zeros((length+1,length+1))
        for i in range(length+1):
            for j in range(i+1,length+1):
                self.time_matrix[i,j] = self.traj_planner(self.pos_array[i],self.pos_array[j])
        self.time_matrix += self.time_matrix.T
        print(self.time_matrix)

        vertex = []
        s = 0
        for i in range(length+1):
            if i != s:
                vertex.append(i)
 
        # store minimum weight Hamiltonian Cycle
        min_path = maxsize
        next_permutation=permutations(vertex)
        pathlength = []
        path = []
        for i in next_permutation:
            # store current Path weight(cost)
            current_pathlength = 0
 
        # compute current path weight
            k = s
            for j in i:
                current_pathlength += self.time_matrix[k,j]
                k = j
            current_pathlength += self.time_matrix[k,s]
 
            # update minimum
            pathlength.append(current_pathlength)
            path.append(i)
        min_path = np.argmin(np.array(pathlength))
        path = list(path[min_path])
        path.append(0)
        optimal_path = [self.pos_array[i] for i in path]
        print(optimal_path)
        print("finish")
        return optimal_path
        

    def traj_planner(self,start,end):
        state_min = Navigator.snap_to_grid(self,(-self.plan_horizon, -self.plan_horizon))
        state_max = Navigator.snap_to_grid(self,(self.plan_horizon, self.plan_horizon))
        x_init = Navigator.snap_to_grid(self,(start[0], start[1]))
        x_goal = Navigator.snap_to_grid(self,(end[0], end[1]))
        print(x_goal)

        # problem = navigator.AStar(
        #     state_min,
        #     state_max,
        #     x_init,
        #     x_goal,
        #     self.occupancy,
        #     self.plan_resolution,
        # )
        problem = navigator.FMTStar.FMTStar(
                state_min,
                state_max,
                x_init,
                x_goal,
                300,
                self.occupancy,
                self.gamma,
                self.pts_per_line
            )
        success = problem.solve()
        planned_path = problem.path
        # print(planned_path)
        traj, t = navigator.compute_smoothed_traj(
            planned_path, self.v_des, self.spline_alpha, self.traj_dt
        )
        return t[-1]

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                6,
                self.map_probs,
            )
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    res = rescue()
    res.run()