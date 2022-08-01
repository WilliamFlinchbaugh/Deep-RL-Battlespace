# Deep-RL-Battlespace
Multi-agent Reinforcement Learning in Simulated Aviation Battle Scenarios
Repo: https://github.com/BallpointPenBoi/Deep-RL-Battlespace
 
# Contributors:
William Flinchbaugh (“Project lead”, wrote every bit of the code, I’m the one that wrote these docs, TAMS student), 
 
Shane Forry (Worked on poster, logistics, ideas, REU student), 
 
Anshuman Singhal (Worked on paper a bit, did some research, TAMS student), 
 
Phongsiri Nirachornkul (Worked on graphics a bit, MS in AI), 
 
Max Levine (Worked on Unity game, REU student)
 
# Background of project:
Essentially, CAE contacted UNT a while back about research with a deep reinforcement learning model that could help train pilots in aviation battle scenarios. The funding never went through, so the goal is to create a working model to pitch back to CAE. We want funding from them for a paper. They are also potentially looking for interns.
 
# Where we are at:
At the start of 2022, Rebecca and Mounika created a basic q-learning model with a relatively basic environment. The graphics were only still images from matplotlib.
In Summer 2022, I essentially completely transformed everything. I adapted the old environment to an OpenAI Gym environment (agent-vs-random and agent-vs-pretrained-model) that uses pygame. I trained the one agent using Stable-Baselines3 PPO and DQN. First we trained it against an agent using random choice (it had the choices forward, shoot, to enemy base, and to enemy plane). Then, I took that trained agent and placed it into the blue plane and trained the red plane against it.
 
However, we needed multi-agent reinforcement learning (MARL) to add more planes. So, I turned that Gym environment into a PettingZoo environment. There were issues training the agents using Stable-Baselines3 because it doesn’t really support MARL, but I got it sorta working. I’m using PPO right now, but it takes a long time to train because of the large observation space and the agents don’t seem to really be learning.
 
We’re at the point where basically the environment works as intended, but the behavior after training is not very desirable. We pretty much need a custom Pytorch model that supports multiple agents. Once that’s done and there’s interesting behavior, CAE should be contacted and a paper should be written.
 
There was a small effort on a Unity game (not in repo) because Unity’s ML-Agents seems much better for this application then pygame and PettingZoo. PettingZoo and other MARL frameworks are also still in their early development stages. However, we didn’t make much progress on that and we didn’t want to throw away our current progress. Unity is likely an easier approach overall, though.
 
I started writing a custom MADDPG model based on a tutorial (listed below) and it runs, but doesn’t seem to learn much. It’s a bit janky since DDPG is meant for continuous action spaces. I also created a DQN model for multi-agent that seems to work, but has to train for a really really long time. Since it’s DQN, the agents might not learn to collaborate.
 
I just started using a dueling DDQN model for the agents which seems to actually be giving decent results. It’s essentially just a torch model that I stole from ML with Phil. Link is below.
 
# Steps going forwards:
Custom multi-agent torch model to replace stable baselines (Possibly PPO? DDQN is somewhat working, but actor critic is likely best for collaboration)
 
Larger state space (including friendly plane information)
 
Assigning roles to each agent (defender, attacker, etc.)
 
Realism (More accurate numbers for speeds, angles, etc.)
 
# Branches:
There’s 3 important branches: agent-vs-random, agent-vs-pretrained-model, and multi-agent
 
agent-vs-random is just a single agent training against a random action blue plane
 
agent-vs-pretrained-model has the blue agent make decisions based off of a pre-trained model that was trained against a random choice blue plane (PPO_1 is the pre-trained model)
 
multi-agent uses PettingZoo instead of OpenAI Gym and trains multiple planes (it’s way more complex)
 
# Installation Guide:
We’ve been using Anaconda in Python 3.9.12 to run the code. I recommend installing miniconda3 and then using VSCode with the conda interpreter. Install miniconda3 from here:
 
https://docs.conda.io/en/latest/miniconda.html
 
And then install VSCode from here:
 
https://code.visualstudio.com/
 
You’ll need to install the python extensions in VSCode
 
Then, open an anaconda prompt from start menu and run
 
`conda init powershell`


`code .`

 
That should open up VSCode through anaconda. After that, you should only need to change the interpreter in the bottom right to miniconda3 base.
 
As for dependencies, it gets a bit complicated with PZ, Gym, SB3, and Supersuit. Supersuit is only needed to train with SB3 because it contains the black-death wrapper. The requirements are in the requirements.txt file. To install the dependencies, open a terminal in the git directory and run:
 
`pip install -r requirements.txt`

 
There is an issue with the above command currently because there are dependency issues, so you might need to pip install each package independently. In addition, you might have issues installing SuperSuit:
 
If on Windows, you'll need to install the Visual Studio Build Tools so that you can install tinyscaler (dependency for supersuit): https://visualstudio.microsoft.com/downloads/
 
If on Linux, you just need to install gcc by running:
 
`sudo apt-get install gcc`

 
If you need any help getting the code to run, feel free to email me at WilliamFlinchbaugh@gmail.com or message me on discord: BallpointPen#6113
 
# Papers/Tutorials/Docs:
First off, I did a pretty good job at commenting the code in the PettingZoo environment, so it shouldn’t be too difficult to figure out what’s going on
 
For the PettingZoo environment, the docs and website are super useful:
 
https://github.com/Farama-Foundation/PettingZoo
 
https://www.pettingzoo.ml/#
 
This is the template I used to build the PettingZoo environment (from PZ docs):
 
https://www.pettingzoo.ml/environment_creation#example-custom-parallel-environment 
 
Here’s the paper for PettingZoo:
 
https://arxiv.org/abs/2009.14471
 
For training the PZ environment, the main approach right now for PZ environments is using Stable-Baselines3. Here’s the tutorial that the PettingZoo devs always point to:
 
https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b 
 
The Stable-Baselines3 docs are pretty good for understanding how everything works:
 
https://stable-baselines3.readthedocs.io/en/master/index.html
 
This video outlines MADDPG implementation in Pytorch:
 
https://www.youtube.com/watch?v=tZTQ6S9PfkE
 
MADDPG uses a centralized critic, but decentralized actors which is good for our application
 
It’s important to note that we cannot use centralized actors because there are two teams
 
Here’s the paper that this video covers:
 
https://arxiv.org/pdf/1706.02275.pdf
 
The same guy, Machine Learning with Phil, also made the DQN and Dueling DDQN that I used with MARL. Here are the links to the videos and the code:
 
https://www.youtube.com/watch?v=H9uCYnG3LlE
 
https://www.youtube.com/watch?v=wc-FxNENg9U
 
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch_2020.py
 
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/dueling_ddqn_torch.py
 
 
 
------------------- OLD STUFF -------------------
CAE Meeting Notes: 
Data collection – Initially the data will be collected at the CAE side. In the meantime, the data collection setup process will start at UNT
Going to CAE on Wednesday 2/23 to get familiar with the lab/people and to discuss more about the data to be used/how much there will be/who will collect it
Data transfer from CAE to UNT side will be either through hard disk or OneDrive
Should maintain a level of abstraction for the security purposes. Will be dealing with the problem statement in the subtask manner.
Contract is still not signed but hopeful it will be by the end of this week (2/25) so that we can fully start project
In person meeting pushed back a week due to weather (03/02)
Talked about plan for security issues and found possible solution of using another software to gather data similar to the MACE data
Plans to get first data batch from CAE this week (02/28) and recreate it using another software to have wider team involvement 
Eventually, UNT will generate the data and all of the runs needed for training once we have the hardware in place (contract is finalized) 
 

Links and team meeting notes:
2/14/2022 tutorials:
Train a deep DQN with TF agents: A good tutorial that is easy to implement right away as an interactive collab notebook. Uses deep reinforcement learning on a game where you move the platform either left or right to keep the free moving pole vertical. Uses average return for the evaluation policy. 
https://colab.research.google.com/github/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb#scrollTo=M7-XpPP99Cy7
 
An introduction to reinforcement learning: 
https://towardsdatascience.com/drl-01-a-gentle-introduction-to-deep-reinforcement-learning-405b79866bf4
https://github.com/mswang12/minDQN/blob/main/minDQN.py

Unity ML-agents Toolkit documentation:
https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs/Readme.md

2/21/2022 Tutorials:
Autonomous Aircraft Sequencing and Separation: https://arxiv.org/pdf/1911.08265.pdf (Ganesh)
 
Connect 4 with deep RL and Monte Carlo Tree Search: Has code repo for connect 4 and Tic Tac Toe  that makes it easy to implement the tutorial. Also uses Deep Q learning and adds in Monte Carlo Tree Search for comparison. They both performed very well and found that the most important thing is being the first player, not second. 
https://towardsdatascience.com/deep-reinforcement-learning-and-monte-carlo-tree-search-with-connect-4-ba22a4713e7a (Rebecca)
 
Self-Driving Cab (Manasa) 
 
Creating custom environment for shower temperature: PDF attached (Mounika) 

2/28/2022 Papers:
Autonomous Aircraft Sequencing and Separation: https://arxiv.org/pdf/1911.08265.pdf (Ganesh)
Autonomous Air traffic controller: https://arxiv.org/pdf/1905.01303v1.pdf (Mounika)
 
Instance-based defense against adversarial attacks in Deep RL: Not an easy to implement tutorial, but good for being closer to our problem statement and having more theoretical details. Talks about “Safe Reinforcement Learning” for defense against adversarial attacks. The algorithm they proposed uses a risk function based on how far a state space is from a known state space, which allows earlier detection of adverse events. 
https://www.sciencedirect.com/science/article/abs/pii/S0952197621003626 (Rebecca)
 
Soft Actor-critic DRL for Fault Tolerant Flight Control: https://paperswithcode.com/paper/soft-actor-critic-deep-reinforcement-learning (Bhargav)
 

6/6/2022 Tutorials:
Creating a simple OpenAI Gym environment: (Use part of what we have, rewrite the rest) https://blog.paperspace.com/creating-custom-environments-openai-gym/
Once we have the environment, this tutorial can help us implement: https://www.gocoder.one/blog/rl-tutorial-with-openai-gym
Good tutorial for training the model: https://www.youtube.com/watch?v=cO5g5qLrLSo
Creating a custom environment: https://www.youtube.com/watch?v=bD6V3rcr_54

