import buffer
import q_network
import numpy as np

# Each agent will be part of a network and will have a policy pi
class Agent():
    def __init__(self, lr, buffer_size, input_dims, fc1_dims, fc2_dims, n_actions, checkpoint_file):