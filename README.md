
<p align="center">
	<img src="./assets/logo.png">
</p>

Welcome to 🚀 <b>ReinforceLab</b> 🧠, the ultimate destination for anyone looking to dive deep into the world of reinforcement learning! Our repository is packed with <b>well-thought-out solutions</b> to a wide range of RL environments, as well as <b>optimized solutions</b> that push boundaries. Whether you're a beginner 🔰 looking for easy-to-understand solutions or an experienced researcher 🧑‍💻 looking to take your skills to the next level, ReinforceLab has something for everyone.

Our goal is to create a community of RL enthusiasts 🤝 who can come together to <b>share knowledge, collaborate on projects, and achieve greatness</b>. We're excited to see what you'll achieve with the help of our solutions and resources!

And for the competitive ones, we have a <b>leaderboard</b> 📊 where you can showcase your skills and compete with others in solving the environments with the highest scores. So, get ready to join the ranks of the ReinforceLab elite 🏆 and start your journey to mastering RL today!

## Getting Started
Reinforcelab is distributed as a Python package. To use it, you should clone and install this repository
```
git clone https://github.com/factoredai/ReinforceLab
cd ReinforceLab
pip install -e .
```

Each task contains two solutions:
- #### Reference
  A clean solution, which aims to provide a reference on how to implement and structure your RL code. You can run a reference solution with the following code:
  ```
  python reinforcelab/tasks/<TASK_NAME>/reference/train.py
  ```
- #### Optimized
  A clean, although more intricate solution, which demonstrates the best performance currently obtained by the team. Performance is going to be measured by Average 100-episode reward and convergence time. You can run an optimized solution with the following code:
  ```
  python reinforcelab/tasks/<TASK_NAME>/optimized/train.py
  ```
