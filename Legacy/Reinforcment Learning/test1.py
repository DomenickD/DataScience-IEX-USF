# ----CURRENTLY NOT WORKING----
#test one for learning reinforcment learning
#I am using ChatGPT4 to learn this. 
#I am adopting the "No Copy-Paste" learning method. I will type out every line here.  
#This is a good way of learning how things work and it helps me understand what's going on



#import necessary libraries
import gym
import random 

#AI says this is how you create the CartPole env
env = gym.make('CartPole-v1')

#we use episodes here kind of like epochs (maybe) and steps per episode. 
num_episodes = 1000
max_steps_per_episode = 200

#Now we define learning parameters
learning_rate = 0.1 #this is familiar from machine learning
discount_rate = 0.99 #need to ask ai what this is (remove comment when you do)

#initialize the Q-table - also, what is a Q-table?
action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]
q_table = [[0 for _ in range(action_space_size)] for _ in range(state_space_size)] #what am I doing here?

# define a function to choose actions using an epsilon-greedy policy 
# what is an epison-greedy policy?
def  choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() 
    else: 
        return max(range(action_space_size), key=lambda action: q_table[state][action])

#this next part is the training loop. this part is like all machine learning but we include rewards
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        #we are telling it to choose an action I think
        epsilon = 1 - (episode / num_episodes)
        action = choose_action(state, epsilon)

        #take the action the bot made, "observe the new state and reward"
        next_state, reward, done, _, _ = env.step(action) # the underscore is the throwaway variable. (idk why we are throwing it away) why is this line not working?

        #now we update this q_table
        old_value = q_table[state][action] #what is wrong with this?
        next_max = max(q_table[next_state])
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_rate * next_max)
        q_table[state][action] = new_value

        state = next_state
        total_reward += reward

        if done:
            break

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()


