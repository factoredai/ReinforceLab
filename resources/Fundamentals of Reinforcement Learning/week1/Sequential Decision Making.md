### 1. 
What is the incremental rule (sample average) for action values?

- $Q_{n+1}= Q_n + \frac{1}{n} [R_n + Q_n]$

- $Q_{n+1}= Q_n - \frac{1}{n} [R_n - Q_n]$

- $Q_{n+1}= Q_n + \frac{1}{n} [Q_n]$

- $Q_{n+1}= Q_n + \frac{1}{n} [R_n - Q_n]$

<details>
<summary>Click to reveal answer</summary>

Answer: $Q_{n+1}= Q_n + \frac{1}{n} [R_n - Q_n]$

At each time step the agent moves its prediction in the direction of the error by the step size (here $\frac{1}{n}$).
</details>

---

### 2. 

Equation 2.5 (from the SB textbook, 2nd edition) is a key update rule we will use throughout the Specialization. We discussed this equation extensively in [this video](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning/lecture/XWqhe/estimating-action-values-incrementally). This exercise will give you a better hands-on feel for how it works. 

$q_{n+1} = q_n + \alpha_n [R_n - q_n]$

Given the estimate update in red, what do you think was the value of the step size parameter we used to update the estimate on each time step?

![Image of estimate update](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/E4RJl4cKEemA0BK4cRbOlg_acc93e119b1793005c3be9b36e9295c4_chart-_4_.png?expiry=1679011200000&hmac=tvJBdelwHWtFSjfu-tBtvBpJmeDfEhbFPeElQzdw_pI)

- 1.0
- 1/2
- 1/8
- 1/(t-1)

<details>
<summary>Click to reveal answer</summary>

Answer: $\frac{1}{2}$

We can see that the estimate is updated by about half of what the prediction error is.
</details>

---

### 3.

Equation 2.5 (from the SB textbook, 2nd edition) is a key update rule we will use throughout the Specialization. We discussed this equation extensively in [video](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning/lecture/XWqhe/estimating-action-values-incrementally "video"). This exercise will give you a better hands-on feel for how it works. The blue line is the target that we might estimate with equation 2.5. The red line is our estimate plotted over time.

$$q_{n+1}=q_n+\alpha_n[R_n-q_n]$$

Given the estimate update in red, what do you think was the value of the step size parameter we used to update the estimate on each time step?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1YaT04cKEemVeg5DpI4LqA_1e09c6b42858c4b8e728a9db745a0ec7_chart-_5_.png?expiry=1679011200000&hmac=wK9JQqBwAjxK3PzKcoN7Nsq3Tb0NLk7BjsjplmzjZ3g)

1. 1.0
2. 1/(t-1)
3. 1/2
4. 1/8

<details>
<summary>Click to reveal answer</summary>

Answer: 1/8

We can see that the estimate is updated by $\frac{1}{8}$ of the prediction error at each time step.
</details>

---

### 4.

Question 4: 

Equation 2.5 (from the SB textbook, 2nd edition) is a key update rule we will use throughout the Specialization. We discussed this equation extensively in [video](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning/lecture/XWqhe/estimating-action-values-incrementally). This exercise will give you a better hands-on feel for how it works. 

The blue line is the target that we might estimate with equation 2.5. The red line is our estimate plotted over time.

$$q_{n+1} = q_n + \alpha_n[R_n - q_n]$$

Given the estimate update in red, what do you think was the value of the step size parameter we used to update the estimate on each time step?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/5oQYd4cKEemVeg5DpI4LqA_5023925429e9e1859eacaf528561f8d9_chart-_6_.png?expiry=1679011200000&hmac=1VoGP94Piw2MD3ifU9JgUGc8AJQoWHLtKsqEdj7-Y6E)

- 1/2
- 1.0
- 1/8
- 1/(t-1)

<details>
<summary>Click to reveal answer</summary>

Answer: 1,0

The estimate is updated to what the previous target was.
</details>

---
### 5.

Equation 2.5 (from the SB textbook, 2nd edition) is a key update rule we will use throughout the Specialization. We discussed this equation extensively in [video](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning/lecture/XWqhe/estimating-action-values-incrementally "video"). This exercise will give you a better hands-on feel for how it works. The blue line is the target that we might estimate with equation 2.5. The red line is our estimate plotted over time.

$q_{n+1} = q_n + \alpha_n [R_n - q_n]$ 

Given the estimate update in red, what do you think was the value of the step size parameter we used to update the estimate on each time step?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/-CfvTIcKEemVeg5DpI4LqA_5d2a80708c491acae15e2f42604ef2b4_chart-_7_.png?expiry=1679011200000&hmac=cy8zYylBdhConKVCPVzmblDUw9pewfMR7ktMos5jycg)

- 1.0
- 1/2
- 1/8
- 1/(t-1)

<details>
<summary>Click to reveal answer</summary>

Answer: 1/(t-1)

We can see that the estimate is updated fully to the target initially, and then over time the amount that the estimate updates is reduced. This indicates that our step size is reducing over time
</details>

---
### 6.

What is the exploration/exploitation tradeoff?

- The agent wants to explore to get more accurate estimates of its values. The agent also wants to exploit to get more reward. The agent cannot, however, choose to do both simultaneously.
- The agent wants to explore the environment to learn as much about it as possible about the various actions. That way once it knows every arm’s true value it can choose the best one for the rest of the time.
- The agent wants to maximize the amount of reward it receives over its lifetime. To do so it needs to avoid the action it believes is worst to exploit what it knows about the environment. However to discover which arm is truly worst it needs to explore different actions which potentially will lead it to take the worst action at times.

<details>
<summary>Click to reveal answer</summary>

Answer: 

> The agent wants to explore to get more accurate estimates of its values. The agent also wants to exploit to get more reward. The agent cannot, however, choose to do both simultaneously.

The agent wants to maximize the amount of reward it receives over time, but needs to explore to find the right action.
</details>

---

### 7.

Why did epsilon of 0.1 perform better over 1000 steps than epsilon of 0.01?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/OdNsrHbGEemVOgruE3X9hg_1d72d77ad61d5d80678e869fa6f793d8_download.png?expiry=1679011200000&hmac=Km57n9qIRRHdQ5yxqYc7MntjeTEzvucxnKwYHrppcpk)

- The 0.01 agent did not explore enough. Thus it ended up selecting a suboptimal arm for longer.
- The 0.01 agent explored too much causing the arm to choose a bad action too often.
- Epsilon of 0.1 is the optimal value for epsilon in general.

<details>
<summary>Click to reveal answer</summary>

Answer: 

> The 0.01 agent did not explore enough. Thus it ended up selecting a suboptimal arm for longer.

The agent needs to be able to explore enough to be able to find the best arm to pull over time. Here epsilon of 0.01 does not allow for enough exploration in the time allotted.
</details>

---

### 8.

If exploration is so great why did epsilon of 0.0 (a greedy agent) perform better than epsilon of 0.4?

![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/8VaPyIcMEemNChIBV74tfA_9eaa84f77f36743c871b8cc26b5c2265_quiz-image.png?expiry=1679011200000&hmac=WsJ9VoChYEP_wpYYshPC1yXsLroVGFcrr5Hjc1p7lbA)

- Epsilon of 0.4 doesn’t explore often enough to find the optimal action.
- Epsilon of 0.4 explores too often that it takes many sub-optimal actions causing it to do worse over the long term.
- Epsilon of 0.0 is greedy, thus it will always choose the optimal arm.

<details>
<summary>Click to reveal answer</summary>

Answer: 

> Epsilon of 0.4 explores too often that it takes many sub-optimal actions causing it to do worse over the long term.

While we want to explore to find the best arm, if we explore too much we can spend too much time choosing bad actions even when we know the correct one. In this case the action-value estimates are likely correct, however the policy does not always choose the action with the highest value.
</details>

