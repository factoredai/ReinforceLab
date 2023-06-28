# Peer Graded Assignment: Describe Three MDPs

## Instructions
For this assignment you will get experience thinking about Markov Decision Processes (MDPs) and how to think about them. You will devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as different from each other as possible.

## Review Criteria
You will be graded on each MDP separately. The grading criteria is:

1. That you have described an MDP and that it is different than your other two.
2. That you have described the MDP's states.
3. That you have described the MDP's actions.
4. That you have described the MDP's rewards.

## Example Submission
An example of an MDP could be a self driving car. The states would be all of the sensor readings that car gets at each time step: LIDAR, cameras, the amount of fuel left, current wheel angle, current velocity, gps location. The actions could be accelerate, decelerate, turn wheels left, and turn wheels right. The rewards could be -1 at every time step so that the agent is encouraged to get to the goal as quickly as possible, but -1 billion if it crashes or breaks the law so that it knows not to do that.