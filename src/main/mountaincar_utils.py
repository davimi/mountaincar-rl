import math
import datetime
import numpy as np


def goal_reward(reward_, is_done) -> float:
    return reward_ if not is_done else 10.0

def cubic_approximation_reward(position_, is_done) -> float:
    if not is_done:
        return 30 * math.pow(position_, 3) - 3
    else:
        return 1

def cubic_approximation_reward_flat(position_, is_done) -> float:
    # https://graphsketch.com/?eqn1_color=1&eqn1_eqn=3(x%20%2B%200.5)%20%5E3%20-%203&eqn2_color=1&eqn2_eqn=-3&eqn3_color=3&eqn3_eqn=3sin(3x)&eqn4_color=4&eqn4_eqn=&eqn5_color=5&eqn5_eqn=&eqn6_color=6&eqn6_eqn=&x_min=-1.2&x_max=1.2&y_min=-5&y_max=5.5&x_tick=0.1&y_tick=1&x_label_freq=5&y_label_freq=5&do_grid=0&do_grid=1&bold_labeled_lines=0&bold_labeled_lines=1&line_width=4&image_w=850&image_h=525
    reward_function_y_offset = 3
    if not is_done:
        return max(3 * math.pow(position_ + 0.5, 3) - reward_function_y_offset, -reward_function_y_offset)
    else:
        return 1

def save_model(agent):
    model_save_name = "model-{}.hdf5".format(datetime.datetime.now())
    agent.save_model_weights(model_save_name)

def normalize_position(position, position_max, position_min) -> float:
    return position / (position_max - position_min)

def normalize_speed(speed, max_speed) -> float:
    return speed / max_speed

def normalize_state(state, position_max, position_min, max_speed) -> np.array:
    return [normalize_position(state[0], position_max, position_min), normalize_speed(state[1], max_speed)]
