# DDQN + PER

Python implementation of DDQN + PER  

## References  
* Deep Q-Learning (DQN): https://arxiv.org/pdf/1312.5602.pdf  
* Double DQN: https://arxiv.org/pdf/1509.06461.pdf  
* Prioritized Experience Replay: https://arxiv.org/pdf/1511.05952.pdf


## Required
 * Python 3.6+
 * Keras 2.0.5 (Tensorflow backend)
 * PIL
 * Numpy
 
Here is an agent I trained using this network:  
https://s1.gifyu.com/images/okayspeed.gif


## Recommended Settings
* RMSprop optimizer for gradient descent with gradient clipped to 1
* Anneal epsilon from 1 to 0.1 over 1,000,000 steps (linear annealing is fine)
* MSE Loss
* ConvNet
* 1 episode every 4 frames
* 32 batch size during learning step
* Target network update interval: Between 200-1000 learning steps

#### Convolutional Neural Network Layers
1. Zero Padding + 16 Filters, Stride=(4,4), ReLU activation
2. Zero Padding + 32 Filters, Stride=(2,2), ReLU activation
3. Flatten + Fully connected layer with 256 units, ReLU activation
4. Output layer with unit=(number of actions agent can perform)

### Example Usage
```python
from ddqn import DDQN
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, ZeroPadding2D
from keras.optimizers import RMSprop
from keras import backend as K
K.set_image_dim_ordering('tf')

# Use any environment here. OpenAI's Gym can be used
# Must ensure that model fits the dimensions of data returned by environment
env = some_arbitrary_environment.load()

cols, rows = env.shape()
frames = 4

shape = (cols, rows, frames) # specific to arbitrary environment
num_actions = 7              # specific to arbitrary environment

def create_model():
    model = Sequential()

    # Layer 1
    model.add(ZeroPadding2D(input_shape=shape))
    model.add(Convolution2D(16, 8, strides=(4,4), activation='relu'))

    # Layer 2
    model.add(ZeroPadding2D())
    model.add(Convolution2D(32, 4, strides=(2,2), activation='relu'))

    # Layer 3
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))

    # Output Layer
    model.add(Dense(units=num_actions, activation='linear'))

    # optimizer and loss
    learning_rate = 0.00025
    loss = "mse"
    rmsprop = RMSprop(lr=learning_rate, clipvalue=1)

    model.compile(loss=loss, optimizer=rmsprop)

    return rmsprop, loss, learning_rate, model

optimizer, loss, learning_rate, model = create_model()

ddqn = DDQN(env, model, loss, optimizer, learning_rate, num_actions, 500, shape)

while True:
    show = env.display()
    ddqn.act_and_learn()
    # Environment automatically resets
    # Check ddqn.py for functions which can print statistics here
```
