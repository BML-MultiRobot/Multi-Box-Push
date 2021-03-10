import numpy as np
import pickle


def to_state_id(robot_id, locations, orientations, ball_location, near_ball):
    """ Generates local, discrete state for robot number {robot_id}

        Indicators: near ball (2), ball location (4), y sign (3), on left (2), on right(2), front(2), back (2)
        Total: 256 states. Lots of them. But, we have lots of robots.

        Simpler: near ball (2), ball location (8), y sign (3)
        Total: 48
        """
    robot_location, robot_ori = locations[robot_id], orientations[robot_id]
    near_ball_indicator = int(distance(robot_location, ball_location) < near_ball)
    ball_location_indicator = determine_direction(robot_location, robot_ori, ball_location)
    ball_y_indicator = 0 if ball_location[1] > 0.05 else (1 if ball_location[1] < -.05 else 2)
    features = [ball_location_indicator, near_ball_indicator, ball_y_indicator]
    # features = [ball_location_indicator, ball_y_indicator]
    state_id = 0
    for i, f in enumerate(features):
        state_id += f * (10 ** i)
    return state_id


def determine_direction(location, orientation, target, ball=True):
    """ Return 0, 1, 2, 3 for front, left, back, right """
    orientation = 0
    vector = np.subtract(target, location)
    target_vector = vector / np.linalg.norm(vector)

    ori_vector = np.array([np.cos(orientation), np.sin(orientation)])
    ori_y_vector = np.array([np.cos(orientation + np.pi /2), np.sin(orientation + np.pi / 2)])
    ori_45_vector = np.array([np.cos(orientation + np.pi / 4), np.sin(orientation + np.pi / 4)])
    ori_135_vector = np.array([np.cos(orientation + 3* np.pi / 4), np.sin(orientation + 3 * np.pi / 4)])

    along_x = np.dot(ori_vector, target_vector)
    along_y = np.dot(ori_y_vector, target_vector)
    along_45 = np.dot(ori_45_vector, target_vector)
    along_135 = np.dot(ori_135_vector, target_vector)

    if ball:
        max_dir = np.argmax([abs(along_x), abs(along_y), abs(along_45), abs(along_135)])
        if max_dir == 0:
            return 0 if along_x > 0 else 4
        if max_dir == 1:
            return 2 if along_y > 0 else 6
        if max_dir == 2:
            return 1 if along_45 > 0 else 5
        if max_dir == 3:
            return 3 if along_135 > 0 else 7

    else:
        max_dir = np.argmax([abs(along_x), along_y])
        if max_dir == 0:
            return 0 if along_x > 0 else 2
        if max_dir == 1:
            return 1 if along_y > 0 else 3


def distance(point1, point2):
    """ L2 distance between two points """
    assert len(point1) == len(point2)
    return np.sqrt(np.sum(np.square(np.subtract(point1, point2))))


def save_data(file_path, data):
    """ Save data to specific path """
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)
