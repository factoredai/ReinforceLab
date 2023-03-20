### 1.

The learner and decision maker is the \_\_\_\_\_\_\_.

- Agent
- Environment
- State
- Reward

<details>
<summary>Click to reveal answer</summary>

Answer: Agent

The agent learns by interacting with the environment and making decisions based on the rewards it receives.
</details>

---
### 2.

At each time step the agent takes an \_\_\_\_\_\_\_.

- State
- Action
- Environment
- Reward

<details>
<summary>Click to reveal answer</summary>

Answer: Action

At each time step, the agent selects an action based on its current policy.
</details>

---
### 3.

What equation(s) define $q_\pi(s,a)$ in terms of subsequent rewards?

 - [ ] $q_\pi(s,a)=[G_t|S_t=s, A_t=a]$ 

    where: $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\dots$

 - [ ] $q_\pi(s,a)=\mathbb{E}_\pi[G_t|S_t=s, A_t=a]$ 

    where: $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\dots$
    
 - [ ] $q_\pi(s,a)=\mathbb{E}_\pi[G_t]$

    where: $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\dots$
    
 - [ ] $q_\pi(s,a)=\mathbb{E}_\pi[R_{t+1}|S_t=s, A_t=a]$

 - [ ] $q_\pi(s,a)=\mathbb{E}_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\dots|S_t=s, A_t=a]$

<details>
<summary>Click to reveal answer</summary>

Answer:
- $q_\pi(s,a)=\mathbb{E}_\pi[G_t|S_t=s, A_t=a]$ 

    where: $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\dots$
- $q_\pi(s,a)=\mathbb{E}_\pi[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\dots|S_t=s, A_t=a]$
</details>

---
### 4.
Imagine the agent is learning in an episodic problem. Which of the following is true?

- The number of steps in an episode is always the same.
- The agent takes the same action at each step during an episode.
- The number of steps in an episode is stochastic: each episode can have a different number of steps.

<details>
<summary>Click to reveal answer</summary>

Answer: The number of steps in an episode is stochastic: each episode can have a different number of steps. 

In an episodic problem, the agent interacts with the environment for a finite number of steps (an episode) and then the environment is reset to its initial state, starting a new episode. Each episode can have a different number of steps.

</details>

---
### 5.
If the reward is always +1 what is the sum of the discounted infinite return when $\gamma<1$?

$G_t=\sum_{k=0}^\infty \gamma^k R_{t+k+1}$

- $G_t=1*\gamma^k$
- $G_t=\frac{1}{1-\gamma}$
- $G_t=\frac{\gamma}{1-\gamma}$
- Infinity.

<details>
<summary>Click to reveal answer</summary>

Answer: $G_t=\frac{1}{1-\gamma}$

If the reward is always +1, the return is just the sum of an infinite geometric series. Thus,

$G_t=\sum_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma}$
</details>

---
### 6.
What is the difference between a small gamma (discount factor) and a large gamma?

- With a smaller discount factor the agent is more far-sighted and considers rewards farther into the future.
- With a larger discount factor the agent is more far-sighted and considers rewards farther into the future.
- The size of the discount factor has no effect on the agent.

<details>
<summary>Click to reveal answer</summary>

Answer: With a larger discount factor the agent is more far-sighted and considers rewards farther into the future.

The discount factor determines how much the agent weights future rewards compared to immediate rewards. A larger discount factor means future rewards are weighted more heavily, which makes the agent more far-sighted and future-oriented in its decision making.
</details>

---
### 7.

Question 7

Suppose $\gamma=0.8$ and we observe the following sequence of rewards: $R_1=-3$, $R_2=5$, $R_3=2$, $R_4=7$, and $R_5=1$, with $T=5$. What is $G_0$? 

Hint: Work Backwards and recall that $G_t=R_{t+1}+\gamma G_{t+1}$.

- 11.592
- 6.2736
- -3
- 8.24
- 12

<details>
<summary>Click to reveal answer</summary>

Answer: 6.2736

$G_5 = R_6 = 0$ (since there is no reward after the last time step).

$G_4 = R_5 + \gamma G_5 = 1 + 0.8 \cdot 0 = 1$

$G_3 = R_4 + \gamma G_4 = 7 + 0.8 \cdot 1 = 7.8$

$G_2 = R_3 + \gamma G_3 = 2 + 0.8 \cdot 7.8 = 8.24$

$G_1 = R_2 + \gamma G_2 = 5 + 0.8 \cdot 8.112 = 11.592$

$G_0 = R_1 + \gamma G_1 = -3 + 0.8 \cdot 11.49 = 6,2736$

</details>

---
### 8.

Suppose $\gamma=0.8$ and the reward sequence is $R_1=5$ followed by an infinite sequence of $10$s. What is $G_0$?

- 45
- 55
- 15

<details>
<summary>Click to reveal answer</summary>

Answer: 45

$G_2 = \frac{10}{1 - 0.8} = 50$

$G_1 = 10 + 0.8 \times 50 = 50$

$G_0 = 5 + 0.8 \times 50 = 45$
</details>

---
### 9.

Suppose reinforcement learning is being applied to determine moment-by-moment temperatures and stirring rates for a bioreactor (a large vat of nutrients and bacteria used to produce useful chemicals). The actions in such an application might be target temperatures and target stirring rates that are passed to lower-level control systems that, in turn, directly activate heating elements and motors to attain the targets. The states are likely to be thermocouple and other sensory readings, perhaps filtered and delayed, plus symbolic inputs representing the ingredients in the vat and the target chemical. The rewards might be moment-by-moment measures of the rate at which the useful chemical is produced by the bioreactor. Notice that here each state is a list, or vector, of sensor readings and symbolic inputs, and each action is a vector consisting of a target temperature and a stirring rate. Is this a valid MDP?

- Yes
- No

<details>
<summary>Click to reveal answer</summary>

Answer: Yes

The problem satisfies the Markov property since the future state depends only on the current state and not on the history of states. The other conditions of the MDP are also satisfied: actions can be taken at each time step, rewards are received after each action, and the next state depends only on the current state and action.
</details>

---
### 10.

Consider using reinforcement learning to control the motion of a robot arm in a repetitive pick-and-place task. If we want to learn movements that are fast and smooth, the learning agent will have to control the motors directly and have low-latency information about the current positions and velocities of the mechanical linkages. The actions in this case might be the voltages applied to each motor at each joint, and the states might be the latest readings of joint angles and velocities. The reward might be +1 for each object successfully picked up and placed. To encourage smooth movements, on each time step a small, negative reward can be given as a function of the moment-to-moment “jerkiness” of the motion. Is this a valid MDP?

<details>
<summary>Click to reveal answer:</summary>

Answer: Yes

This is a valid MDP. The states are the latest readings of joint angles and velocities, and the actions are the voltages applied to each motor at each joint. The reward is +1 for each object successfully picked up and placed, and a small, negative reward can be given on each time step as a function of the moment-to-moment "jerkiness" of the motion to encourage smooth movements. 
</details>

---
### 11.

Imagine that you are a vision system. When you are first turned on for the day, an image floods into your camera. You can see lots of things, but not all things. You can't see objects that are occluded, and of course you can't see objects that are behind you. After seeing that first scene, do you have access to the Markov state of the environment? Suppose your camera was broken that day and you received no images at all, all day. Would you have access to the Markov state then?

Which of the following statements is true?

- You have access to the Markov state before and after damage.
- You have access to the Markov state before damage, but you don't have access to the Markov state after damage.
- You don't have access to the Markov state before damage, but you do have access to the Markov state after damage.
- You don't have access to the Markov state before or after damage.

<details>
<summary>Click to reveal answer</summary>

Answer: You have access to the Markov state before and after damage.

The Markov property refers to the property that the current state summarizes all relevant information needed to make the optimal decision. In this scenario, the first image that floods into the camera contains all the relevant information at that point in time, so it has the Markov property. If the camera is broken and no images are received, there is no information about the environment that can be used to make a decision, so it still has the Markov property, but with an impoverished future. In this case, all possible futures are the same (all blank), so nothing needs to be remembered in order to predict them.

</details>

---
### 12.

What does MDP stand for?

- Markov Deterministic Policy

- Markov Decision Process

- Meaningful Decision Process

- Markov Decision Protocol

<details>
<summary>Click to reveal answer</summary>

Answer: Markov Decision Process

MDP stands for Markov Decision Process, which is a mathematical framework used to model decision-making processes, especially in situations where outcomes are partly random and partly under the control of a decision maker.
</details>

---
### 13.
What is the reward hypothesis?

- Ignore rewards and find other signals.

- Goals and purposes can be thought of as the minimization of the expected value of the cumulative sum of rewards received.

- Goals and purposes can be thought of as the maximization of the expected value of the cumulative sum of rewards received.

- Always take the action that gives you the best reward at that point.

<details>
<summary>Click to reveal answer</summary>

Answer: Goals and purposes can be thought of as the maximization of the expected value of the cumulative sum of rewards received.

The reward hypothesis states that all goals and purposes can be thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (reward). This hypothesis forms the basic premise of reinforcement learning. 
</details>

---
### 14.

Imagine an agent is in a maze-like gridworld. You would like the agent to find the goal, as quickly as possible. You give the agent a reward of +1 when it reaches the goal and the discount rate is 1.0 because this is an episodic task. When you run the agent, it finds the goal, but does not seem to care how long it takes to complete each episode. How could you fix this? (Select all that apply)

- [ ] Give the agent a reward of 0 at every time step so it wants to leave.
- [ ] Give the agent a reward of +1 at every time step.
- [ ] Set a discount rate less than 1 and greater than 0, like 0.9.
- [ ] Give the agent -1 at each time step.

<details>
<summary>Click to reveal answer</summary>

The correct answers are:

- Set a discount rate less than 1 and greater than 0, like 0.9.\
  From a given state, the sooner you get the +1 reward, the larger the return. The agent is incentivized to reach the goal faster to maximize expected return. 

- Give the agent -1 at each time step.\
  Giving the agent a negative reward on each time step tells the agent to complete each episode as quickly as possible.

</details>

---
### 15.

When may you want to formulate a problem as episodic?

- When the agent-environment interaction naturally breaks into sequences. Each sequence begins independently of how the episode ended.

- When the agent-environment interaction does not naturally break into sequences. Each new episode begins independently of how the previous episode ended.

<details>
<summary>Click to reveal answer</summary>

The correct answer is:

- When the agent-environment interaction naturally breaks into sequences. Each sequence begins independently of how the episode ended.

</details>

---
### 16.

When may you want to formulate a problem as continuing?

- When the agent-environment interaction does not naturally break into sequences. Each new episode begins independently of how the previous episode ended.

- When the agent-environment interaction naturally breaks into sequences and each sequence begins independently of how the previous sequence ended.

<details>
<summary>Click to reveal answer</summary>

Answer: When the agent-environment interaction does not naturally break into sequences. Each new episode begins independently of how the previous episode ended.
</details>

---
