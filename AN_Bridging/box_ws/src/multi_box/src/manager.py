#! /usr/bin/env python

import rospy
from std_msgs.msg import Int8, String, Int16
import numpy as np
import sys 
import vrep
import time

# VREP_SCENES = [ ('elevated_scene', '/home/jimmy/Documents/Research/AN_Bridging/Sims/box_simulation.ttt')]
                #,
                # ('flat_scene', '/home/jimmy/Documents/Research/AN_Bridging/Sims/box_flat_simulation.ttt')]
                # ('slope_scene', '/home/jimmy/Documents/Research/AN_Bridging/Sims/box_slope_single_simulation.ttt')

VREP_SCENES = [('stigmergic_scene', '/home/jimmy/Documents/Research/AN_Bridging/Sims/stigmergic_simulation1.ttt')]


class Manager:
    def __init__(self):
        # ROS Publishers
        fin = rospy.Publisher('/finished', Int8, queue_size=1)
        report_sim = rospy.Publisher('/simulation', String, queue_size=1)
        starting = rospy.Publisher('/starting', Int16, queue_size=1)

        # ROS Subscriber
        rospy.Subscriber("/restart", Int8, self.receiveStatus, queue_size=1)

        # V-REP Client Start
        vrep.simxFinish(-1)  # clean up the previous stuff
        clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if clientID == -1:
            print("Could not connect to server")
            sys.exit()

        # EPISODE TRACKING
        counter = 0
        while (counter < episodes):
            print("Episode Number ", counter + 1)
            time.sleep(3)

            # Choose simulation randomly from list
            simulation_index = np.random.choice(range(len(VREP_SCENES)))
            sim_name, sim_path = VREP_SCENES[simulation_index]
            msg = String()
            msg.data = sim_name
            report_sim.publish(msg)
            vrep.simxLoadScene(clientID, sim_path, 0, vrep.simx_opmode_blocking)
            time.sleep(2)

            # Start simulation
            r = 1
            while r != 0:
                r = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

            msg = Int16()
            msg.data = counter + 1
            starting.publish(msg)
            self.restart = False

            # Wait for signal to stop
            while not self.restart:
                x = 1 + 1

            # Stop and restart simulation
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
            counter += 1

        # Finished all episodes
        time.sleep(2)
        msg = Int8()
        msg.data = 1
        fin.publish(msg)

    def receiveStatus(self, message):
        self.restart = message.data if message.data == 1 else self.restart


episodes = 1200

if __name__ == "__main__":
    rospy.init_node('Dummy', anonymous=True)
    manager = Manager()

