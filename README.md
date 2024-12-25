# Q-Learning-Car

This project demonstrates how two autonomous cars can learn to navigate an oval racetrack using a reinforcement learning approach. The core idea is to have each car, via a Q-Learning algorithm, attempt to maximize its own progress without crashing. Over the course of many simulated episodes, the cars refine their policies, adjusting their steering and throttle in ways that allow them to stay on the track longer and accumulate more reward.

What makes this setup particularly engaging is that the two cars follow slightly different reward structures. Car A uses a more nuanced “complex” scheme that includes progress incentives and penalties, while the other Car B uses a comparatively “simple” reward function focused on overall distance traveled. By running both cars in parallel on the same track, it becomes easy to see how small variations in reward design can alter learning behaviors and outcomes.

To accomplish this, each car is represented as a sprite on a 2D oval track rendered in PyGame. The track itself is a PNG image that has a black road surface and white borders. The cars each have seven “radar beams” that project outward from the vehicle in distinct angles, scanning ahead for potential collisions or boundaries. These radars form the basis of the car’s perception: if a radar beam extends until it hits a white border or the edge of the map, the distance traveled along that beam is recorded. This information, combined with each car’s current speed and orientation, is fed into a Q-table as the car’s state. The Q-table then associates every possible state with four discrete actions—turn left, turn right, speed up, or slow down.

During training, the two cars operate under an ε-greedy policy, meaning they sometimes pick random actions to explore, and at other times they exploit what they have learned so far. After every action, each car receives a reward based on factors like how far it traveled (distance-based reward), whether it stayed alive (a slight survival bonus), and progress improvements relative to its best past distance. Car A’s reward function, for instance, offers extra nuance by providing additional penalties or rewards if progress is significantly worse or better than before. Car B’s approach is more direct, awarding a small increment whenever it moves forward. Both vehicles, however, rely on the same underlying Q-Learning update, which factors in a learning rate (α), a discount factor (γ), and the maximum future reward of the next state.

As the simulation progresses, each car’s Q-table grows more accurate in predicting the best action for any given situation. Episodes conclude either when the maximum time step is reached or when a car collides with the map boundary. Between these episodes, the algorithm steadily refines the policies, so that after many iterations, both cars learn to make more skillful maneuvers. One car might settle into a smoother driving style with well-timed accelerations and turns, while the other might approach corners or straights differently based on its simpler reward structure.

Throughout training, the code tracks performance metrics such as total reward per episode and the distance traveled by each car. A live plotting system in Matplotlib updates these metrics in real time, showing the running averages over the past 100 episodes. This visual feedback makes it easy to gauge how quickly the cars’ performance is improving. Often, you will see the lines for average distance and average reward gradually trend upwards as the Q-Learning converges toward better driving policies.

From a technical standpoint, this project highlights how sensor information (the radars), basic physics (updating car positions and angles), and RL algorithms (Q-Learning with an ε-greedy policy) can intertwine within a single simulation. The code is intentionally modular: there is a Car class that manages each vehicle’s position, rotation, collision detection, and radar checks; a Q-Learning routine that updates Q-values; and a main loop that orchestrates rendering, event handling, and data collection. If you wish to experiment further, you could swap in a new map image, change the number of radars, try a different reward scheme, or even replace the table-based Q-Learning with a deep neural network approach.

## Plot Analysis

[over here]

<div style="display: flex; justify-content: space-around;">
    <img src="path_to_reward_plot.png" alt="Average Reward" style="width: 45%;"/>
    <img src="path_to_distance_plot.png" alt="Average Distance" style="width: 45%;"/>
</div>


From these two plots for Average Reward and Average Distance over the last 100 episodes we can see both cars making significant progress. Initially, both reward and distance values fluctuate widely (particularly in the very first episodes, where the orange line sometimes drops below 200 on reward and 300 on distance), reflecting the early exploratory phase. As training continues beyond a few thousand episodes, the lines for both cars steadily climb, indicating that they are learning strategies to drive farther without crashing.

Car A (in blue) often stays slightly ahead of Car B in terms of both reward and distance. That difference highlights how the additional complexity in Car A’s reward function gives it a small but consistent advantage. Still, Car B’s simpler reward structure does not prevent it from improving over time; it also maintains an upward trend, just with a bit more volatility and occasionally lower peaks. By the end of training, each car’s average performance is noticeably higher than at the start, with Car A hovering around (and sometimes exceeding) 900–1000 in distance and 800–900 in average reward, and Car B not far behind.
