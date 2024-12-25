# Imports 

import math
import random
import sys
import os
import pygame
from collections import deque
import matplotlib.pyplot as plt

# Screen size
WIDTH = 1920
HEIGHT = 1080

# Car dimensions 
CAR_SIZE_X = 60
CAR_SIZE_Y = 60

# Background color that is not the road so that if the car hits this color that is not black it will be considered a crash
BORDER_COLOR = (255, 255, 255, 255)  



class Car:
    def __init__(self, sprite_path='car.png', start_x=830, start_y=920):

        # Resizing the car
        self.sprite = pygame.image.load(sprite_path).convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # Inital pos (by def is 830,920)
        self.position = [start_x, start_y]
        
        # Initial angles and attributes
        self.angle = 0
        self.speed = 0
        self.speed_set = False

        # Center of the car
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = []
        self.corners = []
        self.alive = True
        self.distance = 0
        self.time = 0

        # 7 radar directions
        self.radar_angles = range(-135, 136, 45)  

    def draw(self, screen):

        # Drawing the car and its radars
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        # Draw radar lines and points
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        # Check if car collides with borders or goes out of bounds
        self.alive = True
        map_width, map_height = game_map.get_size()
        for point in self.corners:
            px = int(point[0])
            py = int(point[1])
            if px < 0 or px >= map_width or py < 0 or py >= map_height:
                self.alive = False
                break
            if game_map.get_at((px, py)) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):

        # Extending radar until it hits a border or max length
        length = 0
        x = int(self.center[0])
        y = int(self.center[1])
        map_width, map_height = game_map.get_size()

        while length < 300:
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

            if x < 0 or x >= map_width or y < 0 or y >= map_height:
                break

            if game_map.get_at((x, y)) == BORDER_COLOR:
                break
            length += 1

        dist = int(math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2))
        self.radars.append([(x, y), dist])

    def update(self, game_map):

        # Setting speed if not set
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Rotate the sprite
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)

        # Update position based on angle and speed
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculating corners for collision
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Checking collisions and update radars
        self.check_collision(game_map)
        self.radars.clear()
        for d in self.radar_angles:
            self.check_radar(d, game_map)

    def get_data(self):

        # Get radar data and other state info
        radars = self.radars
        radar_vals = [0]*len(self.radar_angles)
        for i, r in enumerate(radars):
            radar_vals[i] = int(r[1] / 30)
        spd = int(self.speed)
        ang = int(round(self.angle / 10.0) * 10)
        return tuple(radar_vals + [spd, ang])

    def is_alive(self):

        # Check if the car is still alive
        return self.alive

    def get_reward(self):

        # Simple reward based on distance
        dist_reward = self.distance / (CAR_SIZE_X / 2)
        return dist_reward

    def rotate_center(self, image, angle):

        # Rotate image around its center
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def select_action(Q, state, epsilon):
    
    # Choose action with epsilon-greedy
    if state not in Q:
        Q[state] = [0.0,0.0,0.0,0.0]
    if random.random() < epsilon:
        return random.randint(0,3)  # Random action
    else:
        return Q[state].index(max(Q[state]))  # Best action

def update_q(Q, state, action, reward, next_state, alpha, gamma, done):

    # Update Q-values
    if next_state not in Q:
        Q[next_state] = [0.0,0.0,0.0,0.0]
    old_value = Q[state][action]
    next_max = max(Q[next_state])
    new_value = (1 - alpha)*old_value + alpha*(reward + gamma*next_max*(0 if done else 1))
    Q[state][action] = new_value

def q_learning():

    actions = [0,1,2,3]  # Actions: left, right, slow, speed
    Q_A = {}
    Q_B = {}
    epsilon = 1.0
    epsilon_decay = 0.9999
    epsilon_min = 0.1
    alpha = 0.1
    gamma = 0.9999

    num_episodes = 10000
    max_steps = 30 * 70

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Two Cars Simultaneous Q-Learning")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 30)

    game_map = pygame.image.load('map.png').convert()

    # For tracking stats
    reward_window_A = deque(maxlen=100)
    reward_window_B = deque(maxlen=100)
    dist_window_A = deque(maxlen=100)
    dist_window_B = deque(maxlen=100)

    best_distance_A = 0.0
    best_distance_B = 0.0
    best_route_A = []
    best_route_B = []

    # Plotting
    plt.ion()
    fig, (ax_reward, ax_distance) = plt.subplots(2,1, figsize=(6,8))
    ax_reward.set_title("Average Reward (Last 100)")
    ax_distance.set_title("Average Distance (Last 100)")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Avg Reward")
    ax_distance.set_xlabel("Episode")
    ax_distance.set_ylabel("Avg Distance")

    line_reward_A, = ax_reward.plot([], [], label="Car A (Complex)")
    line_reward_B, = ax_reward.plot([], [], label="Car B (Simple)")
    line_distance_A, = ax_distance.plot([], [], label="Car A (Complex)")
    line_distance_B, = ax_distance.plot([], [], label="Car B (Simple)")

    ax_reward.legend()
    ax_distance.legend()
    plt.tight_layout()

    rewards_A_list = []
    rewards_B_list = []
    dists_A_list = []
    dists_B_list = []

    for episode in range(num_episodes):
        
        # intilizing the cars

        carA = Car(sprite_path='car.png', start_x=830, start_y=920)
        carB = Car(sprite_path='car.png', start_x=900, start_y=920)  # Slightly offset

        carA.update(game_map)
        carB.update(game_map)

        stateA = carA.get_data()
        stateB = carB.get_data()

        last_distance_A = carA.distance
        last_distance_B = carB.distance

        episode_route_A = []
        episode_route_B = []

        total_reward_A = 0
        total_reward_B = 0

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

            # Choosing the actions
            actionA = select_action(Q_A, stateA, epsilon)
            actionB = select_action(Q_B, stateB, epsilon)

            episode_route_A.append((stateA, actionA))
            episode_route_B.append((stateB, actionB))

            # Applying actions for Car A
            if actionA == 0:
                carA.angle += 10
            elif actionA == 1:
                carA.angle -= 10
            elif actionA == 2:
                if (carA.speed - 2 >= 12):
                    carA.speed -= 2
            elif actionA == 3:
                carA.speed += 2

            # Applying actions for Car B
            if actionB == 0:
                carB.angle += 10
            elif actionB == 1:
                carB.angle -= 10
            elif actionB == 2:
                if (carB.speed - 2 >= 12):
                    carB.speed -= 2
            elif actionB == 3:
                carB.speed += 2

            # Updating cars
            carA.update(game_map)
            carB.update(game_map)

            next_stateA = carA.get_data()
            next_stateB = carB.get_data()

            # Calculating rewards
            survival_reward_A = 0.1 if carA.is_alive() else -1.0
            survival_reward_B = 0.1 if carB.is_alive() else -1.0

            dist_reward_A = carA.get_reward()
            dist_reward_B = carB.get_reward()

            # Car A using a non linear reward system 
            if best_distance_A > 0:
                distance_ratio_A = carA.distance / best_distance_A
                if carA.distance > last_distance_A:
                    progress_reward_A = 0.2 if distance_ratio_A > 0.8 else 0.1
                else:
                    progress_reward_A = -0.5 if distance_ratio_A < 0.5 else -0.1
            else:
                progress_reward_A = 0.1 if carA.distance > last_distance_A else -0.1
            last_distance_A = carA.distance

            # Simple reward system for Car B
            progress_reward_B = 0.1 if carB.distance > last_distance_B else -0.1
            last_distance_B = carB.distance

            rewardA = dist_reward_A + survival_reward_A + progress_reward_A
            rewardB = dist_reward_B + survival_reward_B + progress_reward_B

            total_reward_A += rewardA
            total_reward_B += rewardB

            doneA = (not carA.is_alive()) or (step == max_steps - 1)
            doneB = (not carB.is_alive()) or (step == max_steps - 1)

            done = doneA and doneB

            # Updating the Q-tables
            update_q(Q_A, stateA, actionA, rewardA, next_stateA, alpha, gamma, doneA)
            update_q(Q_B, stateB, actionB, rewardB, next_stateB, alpha, gamma, doneB)

            stateA = next_stateA
            stateB = next_stateB

            # Draw everything
            screen.blit(game_map, (0, 0))
            if carA.is_alive():
                carA.draw(screen)
            if carB.is_alive():
                carB.draw(screen)

            info_text = f"Episode: {episode+1}, Step: {step}, Eps: {epsilon:.2f}"
            text_surface = font.render(info_text, True, (0,0,0))
            screen.blit(text_surface, (10,10))

            pygame.display.flip()
            clock.tick(60)

            if done:
                break

        # Decaying the epsilon value 
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Collecting stats
        final_distance_A = carA.distance
        final_distance_B = carB.distance
        reward_window_A.append(total_reward_A)
        reward_window_B.append(total_reward_B)
        dist_window_A.append(final_distance_A)
        dist_window_B.append(final_distance_B)

        avg_reward_A = sum(reward_window_A)/len(reward_window_A)
        avg_reward_B = sum(reward_window_B)/len(reward_window_B)
        avg_dist_A = sum(dist_window_A)/len(dist_window_A)
        avg_dist_B = sum(dist_window_B)/len(dist_window_B)

        rewards_A_list.append(avg_reward_A)
        rewards_B_list.append(avg_reward_B)
        dists_A_list.append(avg_dist_A)
        dists_B_list.append(avg_dist_B)

        # Updating best routes and apply bonuses/penalties
        if final_distance_A > best_distance_A:
            best_distance_A = final_distance_A
            best_route_A = episode_route_A[:]
            bonus_reward = 0.2
            for (s,a) in best_route_A:
                Q_A[s][a] += alpha * bonus_reward
        else:
            if best_distance_A > 0 and final_distance_A < best_distance_A * 0.5:
                penalty = -0.1
                for (s,a) in episode_route_A:
                    Q_A[s][a] += alpha * penalty

        if final_distance_B > best_distance_B:
            best_distance_B = final_distance_B
            best_route_B = episode_route_B[:]
            bonus_reward = 0.2
            for (s,a) in best_route_B:
                Q_B[s][a] += alpha * bonus_reward
        else:
            if best_distance_B > 0 and final_distance_B < best_distance_B * 0.5:
                penalty = -0.1
                for (s,a) in episode_route_B:
                    Q_B[s][a] += alpha * penalty

        # Printing the episode results
        print(f"Episode {episode+1}: CarA Reward: {total_reward_A:.2f}, Dist: {final_distance_A:.2f}, AvgR: {avg_reward_A:.2f}, AvgD: {avg_dist_A:.2f}")
        print(f"             CarB Reward: {total_reward_B:.2f}, Dist: {final_distance_B:.2f}, AvgR: {avg_reward_B:.2f}, AvgD: {avg_dist_B:.2f}")

        # Updating the plots in real
        line_reward_A.set_xdata(range(len(rewards_A_list)))
        line_reward_A.set_ydata(rewards_A_list)
        line_reward_B.set_xdata(range(len(rewards_B_list)))
        line_reward_B.set_ydata(rewards_B_list)
        line_distance_A.set_xdata(range(len(dists_A_list)))
        line_distance_A.set_ydata(dists_A_list)
        line_distance_B.set_xdata(range(len(dists_B_list)))
        line_distance_B.set_ydata(dists_B_list)

        ax_reward.relim()
        ax_reward.autoscale_view()
        ax_distance.relim()
        ax_distance.autoscale_view()

        plt.pause(0.001)

    pygame.quit()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    q_learning()
