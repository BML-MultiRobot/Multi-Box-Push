import numpy as np
import pickle


def to_state_id(num_bots, locations, orientations, ball_location, near_ball):
    """ Generates local, discrete state for robot number {robot_id}

        Indicators: near ball (2), ball location (4), y sign (3), on left (2), on right(2), front(2), back (2)
        Total: 256 states. Lots of them. But, we have lots of robots.

        Simpler: near ball (2), ball location (8), y sign (3)
        Total: 48
        """
    features = {i: [] for i in range(num_bots)}
    state_ids = {}

    location_indicators, allowed = {}, {}
    occupied = [False for _ in range(16)]  # Indicators for different positions around ball
    for robot_id in range(num_bots):
        robot_location, robot_ori = locations[robot_id], orientations[robot_id]
        near_ball_indicator = int(distance(robot_location, ball_location) < near_ball)
        state_location_indicator, actual_location_indicator = determine_direction(robot_location, robot_ori, ball_location)
        occupied[actual_location_indicator] = True
        location_indicators[robot_id] = actual_location_indicator
        ball_y_indicator = 0 if ball_location[1] > 0.05 else (1 if ball_location[1] < -.05 else 2)
        features[robot_id] = [state_location_indicator, near_ball_indicator, ball_y_indicator]

    for robot_id in range(num_bots):
        f = features[robot_id]
        loc = location_indicators[robot_id]
        clockwise = (loc - 1) % 16
        c_clockwise = (loc + 1) % 16
        f.append(occupied[clockwise])
        f.append(occupied[c_clockwise])
        allowed[robot_id] = {'c': not occupied[clockwise], 'cc': not occupied[c_clockwise]}

        state_id = 0
        for i, f in enumerate(f):
            state_id += f * (100 ** i)
        state_ids[robot_id] = state_id

    return state_ids, location_indicators, allowed

def agent_moved(s, s_prime):
    return s % 10000 != s_prime % 10000

def convert_to_rotation_invariant(states, actions):
    sphere_y = states[-1:]
    agent_locations = states[:-1]
    x = agent_locations[::2]
    y = agent_locations[1::2]

    arc_tans = np.arctan2(y, x)
    arc_tans = np.where(arc_tans < 0, arc_tans + 2 * np.pi, arc_tans)
    indices_each_row = np.argsort(arc_tans)

    new_x_indices = indices_each_row * 2
    new_y_indices = indices_each_row * 2 + 1

    new_indices = np.zeros(agent_locations.shape)
    new_indices[::2] = new_x_indices
    new_indices[1::2] = new_y_indices
    new_indices = new_indices.astype(int)

    # These should create new copies of new_states and new_actions. No in-place operations
    new_states = states[new_indices]
    new_states = np.append(new_states, sphere_y)
    new_actions = actions[indices_each_row]

    return new_states, new_actions

def get_rotation_map(locations_relative_to_ball):
    angles = [np.arctan2(tup[1], tup[0]) for tup in locations_relative_to_ball]
    angles = [a + 2 * np.pi if a < 0 else a for a in angles]
    state_place_to_id = np.argsort(angles)
    return state_place_to_id

def determine_direction(location, orientation, target):
    """ Return 0, 1, 2, 3 for front, left, back, right """
    orientation = 0
    vector = np.subtract(location, target)
    target_vector = vector / np.linalg.norm(vector)

    ori_vector = np.array([np.cos(orientation), np.sin(orientation)])
    ori_y_vector = np.array([np.cos(orientation + np.pi /2), np.sin(orientation + np.pi / 2)])
    ori_45_vector = np.array([np.cos(orientation + np.pi / 4), np.sin(orientation + np.pi / 4)])
    ori_135_vector = np.array([np.cos(orientation + 3* np.pi / 4), np.sin(orientation + 3 * np.pi / 4)])

    along_x = np.dot(ori_vector, target_vector)
    along_y = np.dot(ori_y_vector, target_vector)
    along_45 = np.dot(ori_45_vector, target_vector)
    along_135 = np.dot(ori_135_vector, target_vector)

    ori_22_vector = np.array([np.cos(orientation + np.pi / 8), np.sin(orientation + np.pi / 8)])
    ori_67_vector = np.array([np.cos(orientation + 3*np.pi / 8), np.sin(orientation + 3*np.pi / 8)])
    ori_112_vector = np.array([np.cos(orientation + 5*np.pi / 8), np.sin(orientation + 5*np.pi / 8)])
    ori_157_vector = np.array([np.cos(orientation + 7*np.pi / 8), np.sin(orientation + 7*np.pi / 8)])

    along_22 = np.dot(ori_22_vector, target_vector)
    along_67 = np.dot(ori_67_vector, target_vector)
    along_112 = np.dot(ori_112_vector, target_vector)
    along_157 = np.dot(ori_157_vector, target_vector)

    max_dir = np.argmax([abs(along_x), abs(along_y), abs(along_45), abs(along_135),
                         abs(along_22), abs(along_67), abs(along_112), abs(along_157)])

    state_location, actual_location = None, None
    if max_dir == 0:
        state_location = 0 if along_x > 0 else 4
    elif max_dir == 1:
        state_location = 2 if along_y > 0 else 6
    elif max_dir == 2 or max_dir == 4 or max_dir == 5:
        state_location = 1 if along_45 > 0 else 5
    elif max_dir == 3 or max_dir == 6 or max_dir == 7:
        state_location = 3 if along_135 > 0 else 7

    if max_dir == 0:
        actual_location = 0 if along_x > 0 else 8
    elif max_dir == 1:
        actual_location = 4 if along_y > 0 else 12
    elif max_dir == 2:
        actual_location = 2 if along_45 > 0 else 10
    elif max_dir == 3:
        actual_location = 6 if along_135 > 0 else 14
    elif max_dir == 4:
        actual_location = 1 if along_22 > 0 else 9
    elif max_dir == 5:
        actual_location = 3 if along_67 > 0 else 11
    elif max_dir == 6:
        actual_location = 5 if along_112 > 0 else 13
    elif max_dir == 7:
        actual_location = 7 if along_157 > 0 else 15

    # return state_location, actual_location
    return actual_location, actual_location


def distance(point1, point2):
    """ L2 distance between two points """
    assert len(point1) == len(point2)
    return np.sqrt(np.sum(np.square(np.subtract(point1, point2))))


def save_data(file_path, data):
    """ Save data to specific path """
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)


def load_data(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
