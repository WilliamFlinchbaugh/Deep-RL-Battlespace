# Deep-RL-Battlespace
Multi-Agent Reinforcement Learning in Simulated Aviation Battle Scenarios
 
# Contributors:
- William Flinchbaugh (“Project lead”, wrote every bit of the code, I’m the one that wrote these docs, TAMS student)
- Shane Forry (Worked on poster, logistics, ideas, REU student)
- Anshuman Singhal (Worked on paper a bit, did some research, TAMS student), 
 
# Background of project:
CAE contacted UNT a while back about research with a deep reinforcement learning model that could help train pilots in aviation battle scenarios. The funding never went through, so the goal is to create a working model to pitch back to CAE. We want funding for a paper on the environment and model, or just general usage of MARL in this fashion. They are also potentially looking for interns.
 
# Status/Updates:
At the start of 2022, Rebecca and Mounika created a basic q-learning model with a relatively basic environment. The graphics were only still images from matplotlib.
In Summer 2022, I essentially completely transformed everything. I adapted the old environment to an [OpenAI Gym](https://github.com/openai/gym) environment that uses pygame. I trained the one agent using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) PPO and DQN. First we trained it against an agent using random choice (it had the choices forward, shoot, to enemy base, and to enemy plane). Then, I took that trained agent and placed it into the blue plane and trained the red plane against it.
 
However, we wanted to tackle multi-agent reinforcement learning (MARL) to see if we could get planes to collaborate. So, I turned that Gym environment into a [PettingZoo](https://pettingzoo.farama.org/) environment. We had to switch off of SB3 because it does not support MARL.
 
There was a small effort on a Unity game (not in repo) because Unity’s [ML-Agents](https://unity.com/products/machine-learning-agents) seems much better for this application since PettingZoo and other MARL frameworks are also still in their early development stages (PettingZoo was released in 2020). This effort could be continued in the future for a 3D environment with proper physics. In addition, unity already has MARL algorithms for co-op/competitive environments built in.
 
After using some completely decentralized algorithms, I transitioned to looking for strategies to harbor collaboration between teammates, but still have the planes make their own independent decisions. Here are the approaches I could find (there are not many of these approaches yet):
- [Learning to Share (LToS)](https://arxiv.org/pdf/2112.08702.pdf)
- [Actor-Attention-Critic (MAAC)](https://arxiv.org/pdf/1810.02912.pdf)
- [Multi-Agent Deep Deterministic Policy (MADDPG)](https://arxiv.org/abs/1706.02275)

I went with MADDPG because of the simplicity. I referenced two different repos for help on the model:
- https://github.com/shariqiqbal2810/maddpg-pytorch
- https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients
 
After implementing the MADDPG model, we branched off into two different ways of playing. One was the "self-play" approach where both teams were learning against each other, but this didn't give many results. Instead, I created an "instinct agent" or an algorithm for the opposing team that has a fixed policy. See more details below under behavior.

The "completed_model" in the models folder is the finished model that will be shown to CAE. It wins ~80% of games against the instinct teams and seems to display interesting behavior. It can be tested with the evaluate.py file.

# Behavior:
### MADDPG
The Multi-Agent Deep Deterministic Policy Gradient is an off-policy temporal difference (TD) algorithm for multi-agent environments. It works using a critic network which estimates the Q-value from the actions and observations of the agent's teammates. That Q-value recommends actions to the actor network which chooses actions based on only that agent's observations. A diagram is shown below:

![maddpg drawio (2)](https://user-images.githubusercontent.com/65684280/208565295-d1e9f080-af33-4a6f-aa94-f604f21e228a.png)

### Instinct
The instinct agents are just agents controlled by a set policy. First the agent chooses its target by scoring each of the enemies and then choosing the minimum. The score, s, is calculated through the following equation where d is the distance to the enemy and a is the angle of the enemy relative to the agent:
$s = d*\lvert a\rvert$

After choosing the target, it determines the actions between $[-1, 1]$ using the following calculations:
- Speed: Twice the distance of the target divided by the length of the diagonal of the game field
- Turn: 
    - If aiming left of the target, take max of `-angle/max_turn` and -1
    - If aiming right of the target, take min of `-angle/max_turn` and 1
- Shoot:
    - If within 20 degrees and within two thirds of the distance a bullet can travel, there is a 60% chance of shooting
    - Else, don't shoot (-1)
   
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

Then, simply run the main file to start training. You can tweak the hyperparameters or the length of training in the main.py file.
 
If you need any help getting the code to run, feel free to email me at WilliamFlinchbaugh@gmail.com or message me on discord: BallpointPen#6113
 
# Papers/Tutorials/Docs:
I did a pretty good job at commenting the code, so it shouldn’t be too difficult to figure out what’s going on. If you do have any questions, please feel free to contact me!

For the PettingZoo environment, the docs and website are super useful:
- https://github.com/Farama-Foundation/PettingZoo
- https://www.pettingzoo.ml/#

This is the template I used to build the PettingZoo environment (from PZ docs):
- https://www.pettingzoo.ml/environment_creation#example-custom-parallel-environment 
 
Here’s the paper for PettingZoo:
- https://arxiv.org/abs/2009.14471
 
This video perfectly explains the challenges in MARL, specifically decentralized vs. centralized:
- https://www.youtube.com/watch?v=qgb0gyrpiGk

Here's one of my references for the MADDPG model from Machine Learning with Phil:
- https://www.youtube.com/watch?v=tZTQ6S9PfkE
