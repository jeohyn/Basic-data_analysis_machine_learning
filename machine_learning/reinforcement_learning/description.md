#learning Q-learning(dummy. has many errors)

●Frozen Lake : Even if you know the way, ask.
->ask Q-function where to go. Choose the action which gives the max reward.(in mathmatical expression, max Q(state, action, argmax Q(state, action))
Q-function:Q(state, action), return reward.

-->how to get Q-function?

●assume Q in s' exists
*My condition_ my state is s.
when I do action a, I'll go to s'.
when I do action a, I'll get reweard r.
Q in s', Q(s', a') exists.
->How can we express Q(s, a) using Q(s', a')?
Q(s, a)=r+max Q(s', a') (get r when I do action a, and then get reward when I do a'(max Q(s', a')) next time)

R(t)=r(t)+R(t+1)(r(t) : reward at t, R(t+1) : sum of rewear t+1 to n(terminal reward))

●optimal reward
R*(t)=r(t)+max R(t+1)
-> similar to Q(s,a)=r+max Q(s', a')

●summary
For each s, a initialize table entry Q(s,a)<-0
Observe current state s
Do forever : 
-select an action a and execute it
-receive ummediate reward r
-observe the new state s'
-update the table entry for Q(s,a) as follows:
   Q(s,a)<-r+max Q(s',a')
-s<-s'

#Exploit vs Exploration

Exploit : use the rewards that already have
Exploration : find new rewards

●method
1.E-greedy : define the rate of exploration(e)
e=0.1
if rand<e: //explorate
   a=random
else: //exploit
   a=argmax(Q(s,a))

-->if we repeat this a lot, there's no need to explorate
---->use decaying E-greedy

*decaying E-greedy
for i in range(1000)
   e=0.1/(i+1)
   if random(1)<e: //explorate
      a=random
   else: //exploit
      a=argmax(Q(s,a))

2. add random noise
before select where to go, add random noise

a=argmax(Q(s,a)+random_values)
ex)
a=argamx([0.5, 0.6, 0.3, 0.2]+[0.1, 0.2, 0.7, 0.3])
 =argmax([0.6, 0.8, 1.0])
->select third action

-->if we repeat this a lot, there's no need to explorate
---->use decaying random noise

*decaying random noise
for i in range(1000)
   a=argmax(Q(s,a)+random_values/(i+1))

+)compare E-greedy w/ random noise
if we use random noise, there is a high probability of choosing the next highest reward action, if not the highest reward action than using E-greedy.


=>Use exploit and exploration algorithm to select action.

●discounted reward_use to separate different route but same reward
Q(s,a)<-r+Γ*max Q(s', a') (0<Γ<1)
<=>R(t)=r(t)+Γ*r(t+1)+Γ^2*r(t+2)+...+Γ^n-t*r(n)

●codes
##E-greedy
for i in range(num_episodes):
    e=1./((i/100)+1)
    #the Q-Table learning algorithm
    while not done:
        #choose an action by e greedy
        if np.random.rand(1)<e:
            action=env.action_space.sample()
        else:
            action=np.argmax(Q[state, :])

##random noise
for i in range(num_episodes):
    #choose an action by greedily (with noise) picking from Q table
    action=np.argmax(Q[state, :]+np.random.randn(1, env.action_space.n)/(i+1))

##discounted reward
#discount factor
dis=.99

#update Q-Table w/ArithmeticError new knowledge using decay rate
Q[state, action]=reward+dis*np.max(Q[new_state, :])
