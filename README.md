this version is with an overall controller.

We can dynamically allocate the tasks to robots by change the plan in pursuit_ros_overall.py
The first element represents the number of tracking robtos. The second one represents the following robot.

However, there are still some bugs.
I need to publish the initial state by hand. using "rostopic pub ..... /cockroach_state_actual"
The callback method in overall_controller still need to optimize.
The topic mechanism may be not suitable and I 'd like to change it to service.
