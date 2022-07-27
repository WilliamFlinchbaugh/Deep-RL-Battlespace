from pyparsing import dict_of
import battle_v1
import numpy as np
import supersuit as ss
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.agents = agents
        self.n_agents = len(agents)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, rewards, raw_obs_, state_, dones):
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = rewards
        self.terminal_memory[index] = dones

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for i in range(self.n_agents):
            actor_states.append(self.actor_state_memory[i][batch])
            actor_new_states.append(self.actor_new_state_memory[i][batch])
            actions.append(self.actor_action_memory[i][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:@' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:@' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_id, chkpt_dir, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, agent_id+'_actor', chkpt_dir)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, agent_id+'_critic', chkpt_dir)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, agent_id+'_target_actor', chkpt_dir)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, agent_id+'_target_critic', chkpt_dir)

        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
                
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class MADDPG:
    def __init__(self, actor_dims, critic_dims, agents, n_actions, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='checkpoints/'):
        self.n_agents = len(agents)
        self.agents_list = agents
        self.agents = [Agent(actor_dims[i], critic_dims, n_actions, self.n_agents, self.agents_list[i], chkpt_dir, alpha, beta, fc1, fc2, gamma, tau) for i in range(n_agents)]
        self.n_actions = n_actions

    def save_checkpoint(self):
        print(' === Saving Checkpoint ===')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print(' === Loading Checkpoint ===')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = {}
        for i in range(self.n_agents):
            action = self.agents[i].choose_action(raw_obs[i])
            actions[self.agents_list[i]] = action

        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma*critic_value_.clone()
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss.clone())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def dict_to_vec(dictionary, agents):
    if not isinstance(dictionary, dict):
        return dictionary
    arr = []
    for agent in agents:
        arr.append(dictionary[agent])
    return np.array(arr)

if __name__ == '__main__':
    
    T.autograd.set_detect_anomaly(True)

    # Create a new folder for the model
    for i in range(1, 100):
        if not os.path.exists(f'models/checkpoints{i}/'):
            CHECKPOINT_DIR =  f'models/checkpoints{i}/'
            os.makedirs(CHECKPOINT_DIR)
            break

    cf = {
        'n_agents': 2, # Number of planes on each team
        'show': False, # Show visuals
        'hit_base_reward': 3, # Reward value for hitting enemy base
        'hit_plane_reward': 1, # Reward value for hitting enemy plane
        'miss_punishment': 0, # Punishment value for missing a shot
        'die_punishment': 0, # Punishment value for a plane dying
        'fps': 120, # Framerate that the visuals run at
        'force_discrete_action': True
    }

    env = battle_v1.parallel_env(**cf)
    env = ss.black_death_v3(env)
    agents = env.env.possible_agents[:]
    n_agents = len(agents)
    actor_dims = {agent: env.observation_space(agent).shape[0] for agent in agents}
    actor_dims = dict_to_vec(actor_dims, agents)
    critic_dims = sum(actor_dims)

    n_actions = env.action_space(agents[0]).shape[0]
    maddpg_agents = MADDPG(actor_dims, critic_dims, agents, n_actions, chkpt_dir=CHECKPOINT_DIR)
    memory = MultiAgentReplayBuffer(10000000, critic_dims, actor_dims, n_actions, agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 30000
    total_steps = 0
    red_score_hist = []
    blue_score_hist = []
    evaluate = False
    red_best = 0
    blue_best = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        red_score = 0
        blue_score = 0
        while not env.env.env_done:
            if evaluate:
                env.env.show = True
            else:
                env.env.show = False
            actions = maddpg_agents.choose_action(dict_to_vec(obs, agents))
            obs_, rewards, dones, info = env.step(actions)

            obs = dict_to_vec(obs, agents)
            obs_ = dict_to_vec(obs_, agents)
            rewards_vec = dict_to_vec(rewards, agents)
            dones = dict_to_vec(dones, agents)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            actions = dict_to_vec(actions, agents)

            memory.store_transition(obs, state, actions, rewards_vec, obs_, state_, dones)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            red_rew = 0
            blue_rew = 0
            for red_agent in env.env.possible_red:
                red_rew += rewards[red_agent]
            for blue_agent in env.env.possible_blue:
                blue_rew += rewards[blue_agent]

            red_score += red_rew
            blue_score += blue_rew
            total_steps += 1

        red_score_hist.append(red_score)
        blue_score_hist.append(blue_score)
        avg_red = np.mean(red_score_hist[-100:])
        avg_blue = np.mean(blue_score_hist[-100:])
        if not evaluate:
            if avg_red > red_best:
                maddpg_agents.save_checkpoint()
                red_best = avg_red
            if avg_blue > blue_best:
                maddpg_agents.save_checkpoint()
                blue_best = avg_blue
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f'====================\nEpisode: {i}\nAverage Red Score: {avg_red.round(1)}\nAverage Blue Score: {avg_blue.round(1)}')

        
    """
    Need to change all outputs from env.step to a vec and then give that to the algorithm
    Change code to match tutorial code and convert values to vec before sending to transition
    """