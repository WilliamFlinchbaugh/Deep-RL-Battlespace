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
In Summer 2022, I essentially completely transformed everything. I adapted the old environment to an OpenAI Gym environment (agent-vs-random and agent-vs-pretrained-model) that uses pygame. I trained the one agent using Stable-Baselines3 PPO and DQN. First we trained it against an agent using random choice (it had the choices forward, shoot, to enemy base, and to enemy plane). Then, I took that trained agent and placed it into the blue plane and trained the red plane against it.
 
However, we wanted to tackle multi-agent reinforcement learning (MARL) to see if wed could get planes to collaborate. So, I turned that Gym environment into a PettingZoo environment. We had to switch off of SB3 because it does not support MARL.
 
There was a small effort on a Unity game (not in repo) because Unity’s ML-Agents seems much better for this application since PettingZoo and other MARL frameworks are also still in their early development stages (PettingZoo was released in 2020). This effort could be continued in the future for a 3D environment with proper physics.
 
After using some completely decentralized algorithms, I transitioned to looking for strategies to harbor collaboration between teammates, but still have the planes make their own independent decisions. Here are the approaches I could find (there are not many of these approaches yet):
- Learning to Share (LToS): https://arxiv.org/pdf/2112.08702.pdf
- Actor-Attention-Critic (MAAC): https://arxiv.org/pdf/1810.02912.pdf
- Multi-Agent Deep Deterministic Policy (MADDPG): https://arxiv.org/abs/1706.02275

I went with MADDPG because of the simplicity. I referenced two different repos for help on the model:
- https://github.com/shariqiqbal2810/maddpg-pytorch
- https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients
 
After implementing the MADDPG model, I created an "instinct agent" or an algorithm for the opposing team that has a fixed policy. See more details below under behavior.

The "completed_model" in the models folder is the finished model that will be shown to CAE. It wins ~80% of games against the instinct teams and seems to display interesting behavior. It can be tested with the evaluate.py file.

# Behavior:
### MADDPG
The Multi-Agent Deep Deterministic Policy Gradient is an off-policy temporal difference (TD) algorithm for multi-agent environments. It works using a critic network which estimates the Q-value from the actions and observations of the agent's teammates. That Q-value recommends the moves to the actor network which chooses actions based on that agent's observations alone. A diagram is shown below:

### Instinct
The instinct agents are just agents controlled by a set policy. First the agent chooses its target by scoring each of the enemies and then choosing the minimum. The score, s, is calculated through the following equation where d is the distance to the enemy and a is the angle of the enemy relative to the agent:
$s = d*\lvert a\rvert$

After choosing the target, it determines the actions ($a_0, a_1, a_2$), between $[-1, 1]$ using the following calculations:
- Speed: $D$ is the maximum distance, $a_0 = \frac{2d}{D}-1$
- Turn: $T_m$ is the maximum turn for one timestep, 
    $a_1= 
    \begin{cases}
        max(-a\div T_m, -1) & \text{if } a>0\\
        min(-a\div T_m, 1) & \text{if } a\le0
    \end{cases}$
- Shoot: $d_b$ is the distance a bullet can travel, $R\in[-1,1]$,
    $a_2= 
        \begin{cases}
            1 & \text{if } \lvert a\rvert<20 \land d<\frac{2}{3}d_b \land R<0.6 \\
            -1 & \text{otherwise}
        \end{cases}$
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
