import readchar
import gym
from gym.envs.registration import register
import sys, tty, termios
import matplotlib.pyplot as plt


#stochastic world
env=gym.make('FrozenLake-v0')

####learning part
#initialize table with all zeros
Q=np.zeros([env.obersation_space.n, env.action_space.n])
#set learning parameters. numbers of learning
learning_rate=.85
dis=.99
num_episodes=2000

#create lists to conain total rewards and steps per episode
rList=[]
for i in range(num_episodes):
    #Reset environment and get first new observation
    state=env.reset()
    rAll=0
    done=False

    #the Q-Table learning algorithm
    while not done:
        ##random noise
        #choose an action by greedily (with noise) picking from Q table
        action=np.argmax(Q[state, :]+np.random.randn(1, env.action_space.n)/(i+1))
        

        #get new state and reward from environment
        new_stae, reward, done, _=env.step(action)

        #update Q-Table with new knowledge using learning rate_using discounted reward
        Q[state,action]=(1-learning_rate)*Q[state,action]+learning_rate*(reweard+dis*np.max(Q[new_state,:]))

        rAll+=reward
        state=new_state

    rList.append(rAll)




