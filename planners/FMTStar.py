import numpy as np
import heapq as hq
import random

class FMTStar(object):
    def __init__(self, state_min, state_max, x_init, x_goal, num_samples, occupancy, gamma, pts_per_line):
        self.state_min = (0, 0)
        self.state_max = (4, 3)
        for i in range(100):
           print("init:", x_init, "goal:", x_goal)
        self.x_init = x_init
        self.x_goal = x_goal
        self.occupancy = occupancy
        self.num_samples = num_samples
        
        self.path = None
        self.Vopen = []
        self.Vopen_dict = {}
        self.Vclosed = set()
        self.Vunvisited = set()
        self.V = set()
        
        self.gamma = gamma
        self.pts_per_line = pts_per_line
    
        hq.heappush(self.Vopen, (0, self.x_init))
        self.Vopen_dict[self.x_init] = 0
        
        self.sample()
        self.V = self.Vunvisited.union({self.x_init})
        self.E = dict()
        
    def sample(self):
        # random.seed(1)
        self.Vunvisited.add(self.x_goal)
        while len(self.Vunvisited) != self.num_samples:
            # x = random.uniform(self.x_init[0] - 0.5, self.x_init[1] + 0.5)
            # y = random.uniform(self.x_init[1] - 0.5, self.x_init[1] + 0.5)
            x = random.uniform(self.state_min[0], self.state_max[0])
            y = random.uniform(self.state_min[1], self.state_max[1])
            point = (x, y)
            if self.occupancy.is_free(point):
                self.Vunvisited.add(point)
                
    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def is_close(self, point1, point2):
        dis = self.distance(point1, point2)
        return dis < self.gamma * np.sqrt(np.log(self.num_samples) / self.num_samples)
    
    def near(self, x):
        N = set()
        for point in self.V.difference({x}):
            if self.is_close(point, x):
                N.add(point)
        return N
    
    def solve(self):
        z = self.x_init
        cc = 0
        while self.distance(z, self.x_goal) > 1e-1:
            Nz = self.near(z)
            cc += 1
            Vopen_new = set()
            X_near = Nz.intersection(self.Vunvisited)
            cx = 0
            for x in X_near:
                cx += 1
                Nx = self.near(x)
                Y_near = Nx.intersection(set(self.Vopen_dict.keys()))
                minimum = float('inf')
                ymin = x
                for y in Y_near:
                    cost = self.Vopen_dict[y] + self.distance(y, x)
                    if cost < minimum:
                        minimum = cost
                        ymin = y
                linx = np.linspace(ymin[0], x[0], self.pts_per_line)
                liny = np.linspace(ymin[1], x[1], self.pts_per_line)
                collision_free = True
                for point in zip(linx, liny):
                    if self.occupancy.is_free(point) == False:
                        collision_free = False
                        break
                if collision_free:
                    self.E[x] = ymin
                    Vopen_new.add(x)
                    self.Vunvisited.remove(x)
                    hq.heappush(self.Vopen, (minimum, x))
                    self.Vopen_dict[x] = minimum
            self.Vopen_dict.pop(z)
            hq.heappop(self.Vopen)
            self.Vclosed.add(z)
            if len(self.Vopen_dict) == 0:
                return False
            z = self.Vopen[0][1]
        if z != self.x_goal:
            self.E[self.x_goal] = z
        self.reconstruct_path()
        return True

    def reconstruct_path(self):
        self.path = [self.x_goal]
        # print(self.E)
        point = self.x_goal
        for i in range(10):
            print(self.x_goal in self.E)
        # for i in range(100):
        #     print(point, self.x_init)
        print(self.E)
        while self.distance(point, self.x_init) > 1e-6:
            print("Enter while loop", self.E)
            point = self.E[point]
            self.path.append(point)
        self.path = list(reversed(self.path))
        print(self.path)
        # print(self.path)
        # print(len(self.path))
