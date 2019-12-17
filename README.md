# Reinforcement learning agent

My first shot at a reinforcement learning agent.  
Build with pipenv. Requires Python 3.6.

## Approach

Try to keep all rewards on the same scale, since visualizations of models will be better.  

Tried to incorporate the velocity of the car in the reward function, but this caused the agent to just swing back and forth.  
At the top of the hill, the car slows down, so it would turn back around before reaching the goal.

Inspired by: https://keon.io/deep-q-learning/
