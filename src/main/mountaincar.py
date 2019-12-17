import gym
import rl
from mountaincar_utils import *

import numpy as np

env_max_pos = 0.6
env_min_pos = -1.2
env_abs_max_speed = 0.07
env_goal_pos = 0.5

if __name__ == "__main__":

    episodes = 500
    episode_step_size = 1000
    batch_size = int(episode_step_size / 2)

    # environment setup
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = episode_step_size + 1 # workaround for hardcoded 200 steps per episode wtf
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n


    # agent setup
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    agent = rl.QLearningAgent(state_size, action_size, epsilon_decay=0.995)

    done = False

    for e in range(episodes):
        state = env.reset()

        for i in range(episode_step_size):
            #env.render()

            state = normalize_state(state,env_max_pos, env_min_pos, env_abs_max_speed)
            state = np.reshape(state, [1, state_size])

            action = agent.choose_action(state)

            next_state, _, done, _ = env.step(action)

            position = next_state[0]

            # print("Done: {}, reward: {}, step: {}".format(done, reward,i))
            # print("State: [position, velocity]: " + str(state))

            reward = cubic_approximation_reward_flat(position, done)

            agent.remember(state, action, reward, np.reshape(next_state, [1, state_size]), done)

            state = next_state

            if i % batch_size == 0 and i != 0:
                agent.learn_from_past(batch_size)

            if done:
                agent.learn_from_past(batch_size)
                break

        print("End of episode: {}/{}, epsilon: {}".format(e, episodes, round(agent.epsilon, 4)))


    save_model(agent)

    env.close()
