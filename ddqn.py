import utils
import annealer
import numpy as np
from random import uniform
from replay_memory import PER
from keras.models import model_from_json
from collections import deque

class DDQN(object):
    def __init__(self, environment,
                       model, 
                       loss,
                       optimizer,
                       learning_rate, 
                       num_actions, 
                       target_update_interval, 
                       shape,
                       gamma=0.99, 
                       batch_size=32, 
                       memory_size=10000):
        self.env = environment
        self.model = model
        self.target = None
        self.loss = loss
        self.gamma = gamma
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.memory_size = memory_size
        self.target_update_interval = target_update_interval
        self.memory = PER(self.memory_size)
        self.cols, self.rows, self.frames = shape
        self.state = np.zeros((self.frames, self.cols, self.rows))
        self.learning_steps = 0

        self.avg_q_q = deque()
        self.avg_reward_q = deque()
        self.running_avg_q = deque()
        self.epsilon = annealer.Linear_Anneal(1, 0.1, 80000)
        self.init_state()
        self.load_target_weights()

    def avg_q_add(self, val):
        self.avg_q_q.append(val)      

    def avg_q(self):
        size = len(self.avg_q_q)
        return sum(self.avg_q_q)/size if size > 0 else 0

    def reset_stats(self):
        self.avg_q_q = deque()
        self.avg_reward_q = deque()

    def running(self):
        self.running_avg_q.append(self.avg_reward())
        if len(self.running_avg_q) > 50:
            self.running_avg_q.popleft()

        size = len(self.running_avg_q)
        return sum(self.running_avg_q)/size if size > 0 else 0     

    def avg_reward(self):
        size = len(self.avg_reward_q)
        return sum(self.avg_reward_q)/size if size > 0 else 0        

    def predict(self, X, target_network=False):
        NN = self.target if target_network else self.model
        y = NN.predict(X, batch_size=1)[0]
        A = np.argmax(y)
        return y, A

    def update_target(self, filename="model"):
        if self.env.frame_count % self.target_update_interval == 0 and self.memory.is_full():
            self.load_target_weights(filename)

    def load_target_weights(self, filename="model"):
        utils.save_model(self.model, filename+".json", filename+".h5")
        self.target = utils.load_model(filename+".json", filename+".h5", self.optimizer, self.loss, self.learning_rate, 1)

    def init_state(self):
        im = utils.rgb2gray(self.env.get_image()).T
        for i in range(self.frames):
            self.state[i] = im

    def get_updated_state(self, im):
        tmp = np.array(self.state, dtype=np.uint8)
        self.state[0] = utils.rgb2gray(im).T
        for i in range(1, self.frames):
            self.state[i] = np.array(tmp[i-1], dtype=np.uint8)
        
        return np.array(self.state, dtype=np.uint8).reshape((1, self.cols, self.rows, self.frames))

    def perform_action(self, action):
        if uniform(0, 1) < self.epsilon.value:
            action = np.random.randint(0, self.num_actions) 
        self.env.perform_action(action)
        return action

    def get_reward(self):
        reward = self.env.reward()
        self.avg_reward_q.append(reward)
        return reward

    def transition(self, state):
        """ Returns [S, A, R, S', end] 
            S   - start state
            A   - action to perform in S
            R   - reward for ending up in S' after performing A in S
            S'  - state ended up in after performing A in S
            end - indicates if S' is a terminal state       """
        # Get S and perform A in S
        start_state = self.get_updated_state(self.env.get_image())
        y, action = self.predict(start_state) 

        # Perform A in S
        action = self.perform_action(action)

        # Get new state after performing action
        end_state = self.get_updated_state(self.env.get_image())

        # Check reward in new state
        reward = self.get_reward()

        # Check if new state is terminal (auto resets environment)
        terminal = self.env.terminal()

        if terminal:
            self.init_state()

        # Return the transition
        return [start_state, action, reward, end_state, terminal]
    
    def act_and_learn(self):
        e = self.transition(self.state)
        self.memory.add(e, self.compute_TDerror(e))  
      
        if self.memory.is_full():
            self.epsilon.update()
            batch = self.memory.batch(self.batch_size)
            self.learn( batch )
            for item in batch:
                self.memory.add(item, self.compute_TDerror(item))

        self.update_target() 

    def learn_single(self, S, A, R, S_p, terminal, fit=True):
        # predict Q-values for starting state using the main network
        y, _ = self.predict(S)
        y_old = np.array(y)

        # predict best action in ending state using the main network
        _, A_p = self.predict(S_p)

        # predict Q-values for ending state using the target network
        y_target, _ = self.predict(S_p, target_network=True)

        if fit:
            self.avg_q_add(y[A])

        # update Q[S, A]
        if terminal:
            y[A] = R
        else:
            y[A] = R + self.gamma * y_target[A_p]

        return y_old, y

    def compute_TDerror(self, transition):
        S, A, R, S_p, terminal = transition
        y_old, y = self.learn_single(S, A, R, S_p, terminal)
        TDerror = np.abs(y_old[A]-y[A])
        return TDerror      

    def learn(self, batch):
        self.learning_steps += self.batch_size
        for S, A, R, S_p, terminal in batch:
            _, y = self.learn_single(S, A, R, S_p, terminal)
            self.fit(S, y)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.reshape(1, self.num_actions), verbose=0, batch_size=1)
