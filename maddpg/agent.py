from networks import ActorNetwork, CriticNetwork

class NetworkedAgent:
    def __init__(self, agent_list, n_actions, obs_len, name, tau=0.01, batch_size=128):
        self.agent_list = agent_list
        self.n_action = n_actions
        self.obs_len = obs_len
        self.name = name
        self.tau = tau
        self.batch_size = batch_size
        self.timestep = 0

        self.actor = ActorNetwork(obs_len, n_actions, name=f'actor_{name}')
        self.target_actor = ActorNetwork(obs_len, n_actions, name=f'target_actor_{name}')
        self.critic = CriticNetwork(obs_len, n_actions, name=f'critic_{name}')
        self.target_critic = CriticNetwork(obs_len, n_actions, name=f'target_critic_{name}')

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise
        return action.cpu().detach().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)

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