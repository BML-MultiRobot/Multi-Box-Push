#! /usr/bin/env python

from task import Task, unitVector, dot, vector
from task import distance as dist
import numpy as np

class HierarchicalController():
    def __init__(self):
        """ Don't mess with these parameters unless you know for certain the box and robot are same size """
        self.travel_gain = 2.5  
        self.align_gain = 3
        self.rotate_gain = 3
        self.x_contact = 0
        self.contact = {'left': .6, 'right': -.6}
    
    def getPrimitive(self, s, a):
        # given state, action description and goal, return action representing left/right frequencies
        s = np.array(s).ravel()
        goal_angles, align_y_angles, cross_angles, left_angle, right_angle = self.getAngles(s)
        theta, phi = goal_angles
        alpha, beta, from_align = align_y_angles
        goal1, goal2 = cross_angles

        action = None
        if a == "APPROACH":
            action = [self.travel_gain * phi, self.travel_gain * theta]
        if a == "ANGLE_TOWARDS":
            action = [self.rotate_gain * np.cos(theta), -self.rotate_gain * np.cos(theta)]
        if a == "MOVE_BACK":
            action = [-self.travel_gain, -self.travel_gain]
        if a == "ALIGN_Y":
            action = [self.align_gain * beta * from_align, self.align_gain * alpha  * from_align]
        if a == "PUSH_IN" or a == "CROSS":
            action = [self.travel_gain * goal2, self.travel_gain * goal1]
        if a == 'PUSH_LEFT':
            action = [self.travel_gain * left_angle[1], self.travel_gain * left_angle[0]]
        if a == 'PUSH_RIGHT':
            action = [self.travel_gain * right_angle[1], self.travel_gain * right_angle[0]]    
        if a == 'ANGLE_TOWARDS_GOAL':
            action = [self.rotate_gain * np.cos(goal1), -self.rotate_gain * np.cos(goal1)] 
        return action
    
    def getAngles(self, s):
        s = s.ravel()

        relative_y = s[0]
        relative_x = -s[1]
        """if self.primitive == 'PUSH_IN_HOLE' or self.primitive == 'REORIENT' or self.primitive == 'PUSH_TOWARDS' or self.primitive == 'SLOPE_PUSH':
            relative_y = goal[0]
            relative_x = -goal[1]
        if self.primitive == 'CROSS':
            relative_y = s[4]
            relative_x = -s[5]"""
        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0 # since we want to map -pi to pi
        theta = np.arctan(relative_y/relative_x) + buff 
        phi = -np.pi - theta if theta < 0 else np.pi - theta  
        goal_angles = (theta, phi)

        # NOTE: Depending on the primitive, these all reference the box and some otherpoint past it as well 
        box_from_hole = s[:2] - s[4:6]
        hole = s[4:6]
        aligned = hole - dot(hole, unitVector(box_from_hole)) * unitVector(box_from_hole)
        relative_x = -aligned[1]
        relative_y = aligned[0]
        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0 # since we want to map -pi to pi
        alpha = np.arctan(relative_y/relative_x) + buff 
        beta = -np.pi - alpha if alpha < 0 else np.pi - alpha 
        align_y_angles = (alpha, beta, dist(aligned, np.zeros(2)))

        relative_y = s[4]
        relative_x = -s[5]
        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0 # since we want to map -pi to pi
        goal1 = np.arctan(relative_y/relative_x) + buff 
        goal2 = -np.pi - goal1 if goal1 < 0 else np.pi - goal1
        cross_angles = (goal1, goal2)

        pos = s[:2]
        psi = s[3]
        goal_relative_to_box = np.array([self.x_contact, self.contact['left']])
        rotation_matrix = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        home = pos + rotation_matrix.dot(goal_relative_to_box)
        relative_y = home[0]
        relative_x = -home[1]
        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0 # since we want to map -pi to pi
        alpha = np.arctan(relative_y/relative_x) + buff 
        beta = -np.pi - alpha if alpha < 0 else np.pi - alpha 
        left_angle = (alpha, beta)

        goal_relative_to_box = np.array([self.x_contact, self.contact['right']])
        home = pos + rotation_matrix.dot(goal_relative_to_box)
        relative_y = home[0]
        relative_x = -home[1]
        buff = (-np.pi if relative_y < 0 else np.pi) if relative_x < 0 else 0 # since we want to map -pi to pi
        alpha = np.arctan(relative_y/relative_x) + buff 
        beta = -np.pi - alpha if alpha < 0 else np.pi - alpha 
        right_angle = (alpha, beta)

        return goal_angles, align_y_angles, cross_angles, left_angle, right_angle

    def isValidAction(self, s, a):
        return not self.checkConditions(s.ravel(), a)

    def checkConditions(self, full_state, a, complete = True):
        # given self.prev['A'] and state (unraveled already), check that we've sufficiently executed primitive
        if a == None:
            return True
        s = np.array(full_state).ravel()
        goal_angles, align_y_angles, cross_angles, left_angle, right_angle = self.getAngles(s)
        theta, phi = goal_angles
        alpha, beta, to_align = align_y_angles
        goal1, goal2 = cross_angles

        if a == "ANGLE_TOWARDS":
            return abs(theta - np.pi/2) < 5e-2
        if a == "ANGLE_TOWARDS_GOAL":
            return abs(goal1 - np.pi/2) < 5e-2
        if a == "ALIGN_Y":
            return to_align < .1
        if a == "APPROACH":
            return dist(s[:2], np.zeros(2)) < .7
        if a == 'PUSH_IN':
            return False
        return False

    def feature_2_task_state(self, feature):
        return np.hstack((feature[:2], feature[3:7], feature[8:]))
