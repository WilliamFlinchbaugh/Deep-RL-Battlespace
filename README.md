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
 
However, we needed multi-agent reinforcement learning (MARL) to add more planes. So, I turned that Gym environment into a PettingZoo environment. There were issues training the agents using Stable-Baselines3 because it doesn’t really support decentralized MARL.
 
There was a small effort on a Unity game (not in repo) because Unity’s ML-Agents seems much better for this application then pygame and PettingZoo. PettingZoo and other MARL frameworks are also still in their early development stages. However, we didn’t make much progress on that and we didn’t want to throw away our current progress. Unity is likely an easier approach overall, though.
 
I implemented DQN, dueling DDQN, and PPO models for MARL training. The agents seem to be giving decent results, although we haven’t done any hyperparameter tuning. The models are Pytorch models that I stole from Machine Learning with Phil. Each agent has its networks which means it is completely decentralized. DQN is incredibly slow and not great, the dueling DDQN gives good results and runs quickly, but takes longer to converge, and the PPO takes a long time to run but converges much quicker.
 
I made the process for training and evaluating incredibly simple now. You literally just go into the run_algorithms.py, pick whichever algorithm you want, and hit run. You can change the number of games that it will run for. You can change the reward values and stuff from that file too. We could try more algorithms, however the PPO seems to work well for now. The next step is likely a semi-centralized system where the agents on the same team share rewards and can ‘view’ the other teammates’ observations. This will incentivize and allow for collaboration between agents. As of right now, the models are all completely decentralized with no ‘sharing’.
 
# Steps going forwards:
A semi-centralized model where the agents on each team share rewards in order to promote collaboration
 
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
 
To install the dependencies for this project, run the following command:
 
`pip install -r requirements.txt`

 
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
 
Machine Learning with Phil made the DQN, Dueling DDQN, and PPO that I used to create multiple decentralized agents. Here are the links to his channel and the repo that has the algorithms I used:
 
https://www.youtube.com/watch?v=H9uCYnG3LlE
 
https://www.youtube.com/watch?v=wc-FxNENg9U
 
https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning
 
This video has a perfect explanation of decentralized vs centralized MARL. This also can explain why there’s issues with our current decentralized approach:
 
https://www.youtube.com/watch?v=qgb0gyrpiGk
