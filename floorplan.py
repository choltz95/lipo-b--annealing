import cvxpy as cp
from cvxpy import Variable, Constant, Minimize, Problem
import numpy as np
import placedb
import pylab
import math
import joblib

class Net(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 1.0
    def __init__(self, moduleidxs, idx):
        self.moduleidxs = moduleidxs
        self.idx = idx
        self.U_x = Variable(nonneg=True)
        self.L_x = Variable(nonneg=True)
        self.U_y = Variable(nonneg=True)
        self.L_y = Variable(nonneg=True)

class Box(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 1.0
    def __init__(self, width, height, initialx=0.0, initialy=0.0, initialr=0, idx=0, pl=True, r=False, terminal=False, min_area=None):
        self.min_area = min_area
        self.h = Constant(width)
        self.w = Constant(height)
        self.terminal = terminal
        self.pl = pl
        self.r = r
        
        if pl:
            self.x = Variable(nonneg=True)
            self.y = Variable(nonneg=True)
            self.x.value = initialx
            self.y.value = initialy
        else:
            self.x = Constant(initialx)
            self.y = Constant(initialy)        
        if r:
            self.r = Variable(boolean=True)
            self.r.value = initialr
        else:
            self.r=Constant(initialr)
        self.idx = idx
        self.netidxs = []

    @property
    def position(self):
        return (np.round(self.x.value,2), np.round(self.y.value,2))

    @property
    def size(self):
        return (np.round(self.w.value,2), np.round(self.h.value,2))
    
    @property
    def center(self):
        return self.x + self.r*self.h/2 + (1-self.r)*self.w/2, self.y + self.r*self.w/2 + (1-self.r)*self.h/2

    @property
    def left(self):
        return self.x 

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y

    @property
    def top(self):
        return self.y + self.h

    @property
    def rotation(self):
        return self.r.value

class FloorPlan(object):
    MARGIN = 0.0
    ASPECT_RATIO = 5.0
    def __init__(self, boxes, nets, boundary_W=100, boundary_H=100, max_seconds=10, num_cores=1):
        self.boxes = boxes
        self.nets = nets
        self.num_nodes = len(boxes)
        
        self.boundary_W = Constant(boundary_W)
        self.boundary_H = Constant(boundary_H)
        #self.height = Variable(pos=True)
        #self.width = Variable(pos=True)
        self.height = Constant(boundary_H)
        self.width = Constant(boundary_W)
        
        self.p = cp.Variable(shape=(self.num_nodes,self.num_nodes), boolean=True)
        self.q = cp.Variable(shape=(self.num_nodes,self.num_nodes), boolean=True)
        self.horizontal_orderings = []
        self.vertical_orderings = []
        
        self.max_seconds = max_seconds
        self.num_cores = num_cores

    @property
    def size(self):
        return (np.round(self.width.value,2), np.round(self.height.value,2))

    # Return constraints for the ordering.
    @staticmethod
    def _order(boxes, horizontal):
        if len(boxes) == 0: return
        constraints = []
        curr = boxes[0]
        for box in boxes[1:]:
            if horizontal:
                constraints.append(curr.right + FloorPlan.MARGIN <= box.left)
            else:
                constraints.append(curr.top + FloorPlan.MARGIN <= box.bottom)
            curr = box
        return constraints

    # Compute minimum perimeter layout.
    def layout(self):
        constraints = []
        for box in self.boxes:
            if (not box.pl):
                continue
            # Enforce that boxes lie in bounding box. 
            #constraints += [self.height <= self.boundary_H,
            #                self.width <=self.boundary_W]
            constraints += [box.x >= FloorPlan.MARGIN,
                box.x + box.r*box.h + (1-box.r)*box.w + FloorPlan.MARGIN <= self.width]
            constraints += [box.y >= FloorPlan.MARGIN,
                            box.y + box.r*box.w + (1-box.r)*box.h + FloorPlan.MARGIN <= self.height]
            
            # Enforce aspect ratios.
            #constraints += [(1/box.ASPECT_RATIO)*box.height <= box.width,
            #                box.width <= box.ASPECT_RATIO*box.height]
            # Enforce minimum area
            #constraints += [
            #    geo_mean(vstack([box.width, box.height])) >= math.sqrt(box.min_area)
            #]
            
        # wirelength minimization
        for net in self.nets:
            if len(net.moduleidxs) <= 1: continue
            modules = [self.boxes[i].center for i in net.moduleidxs]
            mx, my = list(zip(*modules))
            constraints += [net.L_x <= x for x in mx]
            constraints += [net.U_x >= x for x in mx]
            constraints += [net.L_y <= y for y in my]
            constraints += [net.U_y >= y for y in my]
        
        # nonoverlap constraintss
        for i in range(len(self.boxes)):
            for j in range(i+1,len(self.boxes)):
                b_i = self.boxes[i]
                b_j = self.boxes[j]
                
                if (not b_i.pl) or (not b_j.pl):
                    continue
                
                x_i, y_i = b_i.x, b_i.y
                w_i, h_i = b_i.w, b_i.h
                r_i = b_i.r
                
                x_j, y_j = b_j.x, b_j.y
                w_j, h_j = b_j.w, b_j.h
                r_j = b_j.r
                                
                constraints += [
                    x_i + r_i*h_i + (1-r_i)*w_i <= x_j + self.boundary_W*(self.p[i,j] + self.q[i,j]) - FloorPlan.MARGIN,
                    y_i + r_i*w_i + (1-r_i)*h_i <= y_j + self.boundary_H*(1 + self.p[i,j] - self.q[i,j]) - FloorPlan.MARGIN,
                    x_i - r_j*h_j - (1-r_j)*w_j >= x_j - self.boundary_W*(1 - self.p[i,j] + self.q[i,j]) + FloorPlan.MARGIN,
                    y_i - r_j*w_j - (1-r_j)*h_j >= y_j - self.boundary_H*(2 - self.p[i,j] - self.q[i,j]) + FloorPlan.MARGIN,
                ]

        # Enforce the relative ordering of the boxes.
        for ordering in self.horizontal_orderings:
            constraints += self._order(ordering, True)
        for ordering in self.vertical_orderings:
            constraints += self._order(ordering, False)
            
        #obj = Minimize(2*(self.height + self.width))
        hpwls = [(net.U_x - net.L_x) + (net.U_y - net.L_y) for net in self.nets]
        hpwl = Minimize(cp.sum(hpwls))
        obj = hpwl
        p = Problem(obj, constraints)
        assert p.is_dqcp()
        return p.solve(solver=cp.CBC, warm_start=True, verbose=True, maximumSeconds=self.max_seconds, numberThreads=self.num_cores), constraints

    def verify_constraints(self, constraints):
        return np.array([c.violation() for c in constraints])

    # Show the layout with matplotlib
    def show(self):
        pylab.figure(facecolor='w')
        for k in range(len(self.boxes)):
            box = self.boxes[k]
            x,y = box.position
            if box.rotation:
                h,w = box.size
            else:
                w,h = box.size
                
            pylab.fill([x, x, x + w, x + w],
                       [y, y+h, y+h, y])
            pylab.text(x+.5*w, y+.5*h, "%d" %(k+1))
        
        for k in range(len(self.nets)):
            net = self.nets[k]
            modules = [[p.value for p in self.boxes[i].center] for i in net.moduleidxs]
            mx, my = list(zip(*modules))
            pylab.plot(mx,my, color='gray',alpha=0.25)
        
        x,y = self.size
        pylab.axis([0, x, 0, y])
        pylab.xticks([])
        pylab.yticks([])

        pylab.show()
        
"""
%%time
boxes = [Box(10, 3), Box(5,2), Box(2, 4), Box(1,8), Box(3,10)]
fp = FloorPlan(boxes)
#fp.horizontal_orderings.append( [boxes[0], boxes[2], boxes[4]] )
#fp.horizontal_orderings.append( [boxes[1], boxes[2]] )
#fp.horizontal_orderings.append( [boxes[3], boxes[4]] )
#fp.vertical_orderings.append( [boxes[1], boxes[0], boxes[3]] )
#fp.vertical_orderings.append( [boxes[2], boxes[3]] )
p, c = fp.layout()
fp.show()
violations = fp.verify_constraints(c)
print(fp.height.value, fp.width.value)
print(2*(fp.height.value + fp.width.value))
assert np.all(violations <= 1e-5)
"""