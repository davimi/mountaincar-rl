from mountaincar import *

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns


def show_3D_plot():
    plt.clf()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap=cm.viridis)
    ax.set_xlabel("position")
    ax.set_ylabel("speed")
    ax.set_zlabel("predicted value")

    plt.show()


def show_heatmap(values, xs, ys):
    sns.set()
    df = pd.DataFrame(values, xs, ys)
    ax = sns.heatmap(df)
    ax.set_ylabel("position")
    ax.set_xlabel("speed")

    plt.show()

env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = rl.QLearningAgent(state_size, action_size, epsilon = 0.0, epsilon_min = 0.0)

trained_model = agent.load_model_weights("src/main/resources/model_01.hdf5")

#xs = np.arange(env_min_pos, stop=env_max_pos, step=0.1)
#ys = np.arange(-env_abs_max_speed, stop=env_abs_max_speed, step=0.01)

xs = np.linspace(env_min_pos, env_goal_pos, 20)
ys = np.linspace(-env_abs_max_speed, env_abs_max_speed, 20)
zs = np.empty((len(xs), len(ys)))
actions = np.empty((len(xs), len(ys)))

for x in range(len(xs) - 1 ):
    for y in range(len(ys) - 1):
        reward = cubic_approximation_reward_flat(xs[x], (xs[x] >= env_goal_pos))
        state = normalize_state(np.array([xs[x], ys[y]]), env_max_pos, env_min_pos, env_abs_max_speed)
        state = np.reshape(state, [1, state_size])
        zs[x][y] = agent.predict_value(reward, state)
        actions[x][y] = agent.choose_action(state)

print(zs)
show_heatmap(zs, xs, ys)
#show_heatmap(actions, xs, ys)
#show_3D_plot()
