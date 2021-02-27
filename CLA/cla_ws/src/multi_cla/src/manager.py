#! /usr/bin/env python

import rospy
from std_msgs.msg import Int8, String, Int16
import numpy as np
import sys 
from vrep_util.vrep import *
import time

VREP_SCENES = [('stigmergic_scene', '/home/jimmy/Documents/Research/CLA/sims/kilobot_push.ttt')]


class Manager():
    def __init__(self):
        # ROS Publishers
        fin = rospy.Publisher('/finished', Int8, queue_size=1)
        report_sim = rospy.Publisher('/simulation', String, queue_size=1)
        rospy.Subscriber("/start", Int8, self.receive_ready, queue_size=1)

        # ROS Subscriber
        rospy.Subscriber("/restart", Int8, self.receive_status, queue_size=1)

        # V-REP Client Start
        self.start = False
        simxFinish(-1)  # clean up the previous stuff
        client_id = simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if client_id == -1:
            print("Could not connect to server")
            sys.exit()

        # EPISODE TRACKING
        counter = 0
        while counter < episodes:
            while not self.start:
                x = 1 + 1
            print("Episode Number ", counter + 1)
            time.sleep(3)

            # Choose simulation randomly from list
            simulation_index = np.random.choice(range(len(VREP_SCENES)))
            sim_name, sim_path = VREP_SCENES[simulation_index]
            msg = String()
            msg.data = sim_name
            report_sim.publish(msg)
            simxLoadScene(client_id, sim_path, 0, simx_opmode_blocking)
            time.sleep(2)

            # Start simulation
            r = 1
            while r != 0:
                r = simxStartSimulation(client_id, simx_opmode_oneshot)

            self.start = False
            self.restart = False
            msg = Int16()
            msg.data = counter + 1

            # Wait for signal to stop
            while not self.restart:
                x = 1 + 1

            # Stop and restart simulation
            simxStopSimulation(client_id, simx_opmode_oneshot)
            counter += 1

        # Finished all episodes
        time.sleep(2)
        msg = Int8()
        msg.data = 1
        fin.publish(msg)

    def receive_status(self, message):
        self.restart = message.data if message.data == 1 else self.restart
        return

    def receive_ready(self, message):
        self.start = True
        return


episodes = 300


if __name__ == "__main__":
    rospy.init_node('Dummy', anonymous=True)
    manager = Manager()
