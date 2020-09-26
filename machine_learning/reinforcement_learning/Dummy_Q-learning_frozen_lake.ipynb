import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector): #https://gist.github.com/stober/1943451
    m=np.amax(vector)
    indices=np.nonzero(vector==m)[0]
    return pr.choice(indices)

####Register FrozenLake w/ is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery':False}
    )

env=gym.make('FrozenLake-v3')

####learning part
#initialize table with all zeros
Q=np.zeros([env.obersation_space.n, env.action_space.n])
#set learning parameters. numbers of learning
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
        #if there're same reward, select action randomly(rargmax)
        action=rargmax(Q[state, :])

        #get new state and reward from environment
        new_stae, reward, done, _=env.step(action)

        #update Q-Table with new knowledge using learning rate
        Q[state, action]=reward+np.max(Q[new_state,:])

        rAll+=reward
        state=new_state

    rList.append(rAll)

####print the result
print("Success rate : "+str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()



        
