#! /usr/bin/env python


import numpy as np 
import torch
import rospy
from collections import OrderedDict

from Tasks.hierarchy_MBRL_task import Hierarchy_MBRL_Task
from aa_graphMap_node_simulation import stigmergic_main
from Algs.hierarchical_MBRL import Hierarchical_MBRL

NAME = 'bot'

algs = {
    0: 'SINGLE',
    1: 'STIGMERGIC'
}
ALGORITHM = 0
description = algs[ALGORITHM]
rospy.init_node('Dummy', anonymous=True)

if description == 'SINGLE':
    NAME = 'a_bot'
    agents = OrderedDict({
                #ensure ordering matches ros message
                "a_bot": {"sub": "/state", "pub": "/action"}, #joint action space
            })
    agents["b_bot"] = {"sub": "/state", "pub": "/action2"}

    valPars = {
                'neurons':      (18, 400, 400, 400, 9), # 5 box-related state, 4 goal-related state, 8 action one hot, 1 indicator
                'act':          ['F.leaky_relu','F.leaky_relu', 'F.leaky_relu'],
                'mu':           torch.zeros(18),
                'std':          torch.ones(18),
                'trainMode':    False, # Make sure both value and policy are set to the same thing
                'load':         False, 
                'dual':         False,
                'u_n':          8,
                's_n':          8,
                'dropout':      0
                } 
    valTrain = {
                'batch':        256, #512 used to be...but might be too slow 
                'lr':           3e-4, 
                'noise':        .05,
                'buffer':       10000,
                'gamma':        0,  # Tune the policy below
                'double':       True,
                'one_hot':      True,
                'min_explore':  .2,
                'explore':      .7,  # this was changed to .5 for MB
                'explore_decay': .9998,
                }
    
    policyPars = {
                'neurons':      (10, 200, 200, 200, 8),  # 5 box-related state, 4 goal-related state, 8 controls
                'act':          ['F.leaky_relu', 'F.leaky_relu', 'F.leaky_relu'],
                'mu':           torch.zeros(10),
                'std':          torch.ones(10),
                'trainMode':    False,  # Make sure both value and policy are set to the same thing
                'load':         False, 
                'dual':         False,
                'beta':         16  # boltzmann. Increase for more certainty when making decisions
                } 
    policyTrain = {
                'batch':        128,  # used to be 256
                'lr':           3e-4,
                'buffer':       10000,
                'gamma':        .975,
                'explore':      0,  # Don't change this. Tune the exploration up top
                'double':       True,
                'noise':        0,
                'priority':     False,
                'update_target_network_every': 300,
                'train_every_steps': 1,  # note: ratio of env steps to gradient steps is always 1
                'explore_steps': 5000,
                }
    doubleQPars = {'valPars': policyPars, 'valTrain': policyTrain, 'agents': agents}
    params = {"valPars": valPars, "valTrain": valTrain, 'doubleQ_params': doubleQPars, "agents": agents}
    agent = Hierarchical_MBRL(params, NAME, Hierarchy_MBRL_Task())

if description == 'STIGMERGIC':
    """
    node_data: list of tuples corresponding to: (node coordinates, box_id...-1 if just distance stuff, 
                                                                   non-negative if box candidate)
    inclusions: list of tuples corresponding to (node index, node index)
    exclusions: list of tuples corresponding to (node index, index)
    box: list of box info tuples (index of current node, box_id, height)
    robot: list of tuples (index of current node, robot_id)
    goal: index of the node that is considered goal
    """

    # Basic environment. 1 box. 2 holes (1 invalid). 4 nodes.

    # nodes = [(0, 0, 1, -1), (0, 1, -1, 0), (1, 0, 0, 0), (1, 1, 1, -1)]
    # inclusions = []
    # exclusions = [(1, 2), (0, 3)]
    # box = [(0, 0, 1)]
    # robot = [(0, 0)]
    # goal = 3

    # Basic environment. 2 boxes (1 invalid). 1 hole. 5 nodes.

    # nodes = [(0, 0, 1, -1), (0, 1, 1, -1), (1, 0, 1, -1), (1, 1, 0, 0), (2, 1, 1, -1)]
    # inclusions = []
    # exclusions = [(1, 2), (2, 4), (1, 4)]
    # box = [(1, 0, 1), (2, 0, 3)]
    # robot = [(0, 0)]
    # goal = 4

    # Basic environment. 2 boxes. 1 hole (stack). 5 nodes.

    # nodes = [(0, 0, 1, -1), (0, 1, 1, -1), (1, 0, 1, -1), (1, 1, -1, 0), (2, 1, 1, -1)]
    # inclusions = []
    # exclusions = [(1, 2), (2, 4), (0, 3), (1, 4)]
    # box = [(1, 0, 1), (2, 0, 1)]
    # robot = [(0, 0)]
    # goal = 4

    # Medium Environment. 2 boxes. 2 holes. 9 nodes

    # nodes = [(0, 0, 1, -1), (np.sqrt(2) / 2.0, np.sqrt(2) / 2.0, 1, -1), (np.sqrt(2) / 2.0, -np.sqrt(2) / 2.0, 1, -1),
    #          (2, 0, 0, 0), (3, 0, 1, -1), (3, 1, 1, -1), (3, -1, 1, -1), (4, 0, -2, 0), (5, 0, 1, -1)]
    # inclusions = []
    # exclusions = [(1, 2)]
    # box = [(1, 0, 1), (2, 0, 3)]
    # robot = [(0, 0)]
    # goal = 8

    # V-REP Environment. Note: Make graph generating radius to be 2 for proper map.
    nodes = [[0.8379172682762146, 0.9883939027786255, 5.031623363494873, -1], [0.7651996612548828, 2.364687204360962, 5.044251918792725, -1],
             [2.015198230743408, 0.9147075414657593, 5.044280529022217, -1], [3.715198278427124, 0.9647090435028076, 5.044276714324951, -1],
             [5.1901984214782715, 1.0896940231323242, 4.694263458251953, 0], [6.940199375152588, 1.4646936655044556, 5.044262886047363, -1],
             [7.790197849273682, 2.3646934032440186, 5.044262886047363, -1], [2.0401992797851562, 2.5146865844726562, 5.044251918792725, -1],
             [1.0151994228363037, 3.689687728881836, 5.044251918792725, -1], [2.2651991844177246, 4.239688396453857, 5.044251918792725, -1],
             [4.340200424194336, 4.214687824249268, 5.044251918792725, -1], [3.590200901031494, 3.214689016342163, 5.044251918792725, -1],
             [6.890202522277832, 8.464689254760742, 5.044251918792725, -1], [2.390199661254883, 6.46465539932251, 5.044226169586182, -1],
             [2.665200710296631, 7.939693450927734, 5.044251918792725, -1], [0.6902013421058655, 7.764689922332764, 5.044251918792725, -1],
             [1.515201449394226, 8.614690780639648, 5.044251918792725, -1], [5.390201091766357, 8.389692306518555, 5.044251918792725, -1],
             [4.040201663970947, 6.514691352844238, 5.044251918792725, -1], [7.740201950073242, 6.339690685272217, 5.044251918792725, -1],
             [6.090200901031494, 6.339690685272217, 5.044251918792725, -1], [7.940202236175537, 7.864691257476807, 5.044251918792725, -1],
             [7.865201473236084, 9.689688682556152, 5.044251918792725, -1], [5.115200519561768, 7.2896904945373535, 5.044251918792725, -1],
             [5.865200519561768, 9.714691162109375, 5.044251918792725, -1], [4.165201663970947, 9.514688491821289, 5.044251918792725, -1],
             [2.4652013778686523, 9.639690399169922, 5.044251918792725, -1], [0.7402012348175049, 9.714693069458008, 5.044251918792725, -1],
             [4.090201377868652, 8.114690780639648, 5.044251918792725, -1], [0.6402012705802917, 6.089689254760742, 5.044251918792725, -1],
             [2.4401979446411133, 5.214694499969482, 4.694263458251953, 0]]
    inclusions = [(11, 2)]
    exclusions = []
    box = [[3, 0, 0.35888671875], [13, 0, 0.35888671875]]
    robot = [[2, 0], [8, 1], [14, 2], [19, 3]]
    goal = 6

    """nodes = [(0, 0, 1, -1), (np.sqrt(2) / 2.0, np.sqrt(2) / 2.0, 1, -1), (np.sqrt(2) / 2.0, -np.sqrt(2) / 2.0, 1, -1),  # 0 - 2
             (2, 0, 0, 0), (3, 0, 1, -1), (3, 1, 1, -1), (3, -1, 1, -1), (4, 0, -3, 0), (5, 0, 1, -1), (0, -1.5, 1, -1),  # 3 - 9
             (1.5, -1.5, 1, -1), (4, -2, -2, 0), (5, -1, 1, -1), (6, 0, 1, -1), (5, 1, 1, -1), (6, 1, 1, -1), (3, 2, 1, -1), # 10 - 16
             (0, -2.5, 1, -1), (0, -4, 1, -1), (1.5, -2.5, -8, 0), (1.5, -3.5, 1, -1), (3, -2.5, 1, -1), (3, -3.5, -1, 0),  # 17 - 22
             (3, -4.5, 1, -1)]  # 23
    inclusions = []
    exclusions = [(1, 2), (6, 10), (12, 13), (8, 15), (13, 14), (20, 23)]
    box = [(1, 0, 1), (5, 0, 3), (9, 0, 2)]
    robot = [(0, 0), (6, 1), (23, 2)] #  (10, 2),
    goal = 13"""


    stigmergic_main(nodes, inclusions, exclusions, box, robot, goal)

rospy.spin()
