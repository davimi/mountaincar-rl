import gym
import rl
import numpy as np

if __name__ == "__main__":

    episode_step_size = 5000

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = episode_step_size + 1 # workaround for hardcoded 200 steps per episode wtf
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    state = env.reset()

    agent = rl.QLearningAgent(state_size, action_size, epsilon = 0.0, epsilon_min=0.0)

    trained_model = agent.load_model_weights("src/main/resources/model_02.hdf5")

    done = False

    while not done:
        env.render()
        state = np.reshape(state, [1, state_size])

        action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)

        state = next_state
