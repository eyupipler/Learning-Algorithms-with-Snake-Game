import pygame
import numpy as np
import random
import pickle

pygame.init()

width, height = 640, 480
block_size = 20

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Q-learning Snake Game")

clock = pygame.time.Clock()

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

font = pygame.font.SysFont(None, 35)

# Q-learning
epsilon = 0.2  # Keşif oranı / Explore rate
gamma = 0.9  # İndirim faktörü / Discount factor
alpha = 0.1  # Öğrenme oranı / Learning rate

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# Q-tablosunu oluşturma / Creating Q-table
q_table = {}

def get_state(snake_head, food):
    return (snake_head[0], snake_head[1], food[0], food[1])

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        if state not in q_table:
            q_table[state] = {a: 0 for a in actions}
        return max(q_table[state], key=q_table[state].get)

def get_reward(snake_head, food, game_over):
    if game_over:
        return -100
    elif snake_head == food:
        return 100
    else:
        return -1

def draw_snake(block_size, snake_list):
    for block in snake_list:
        pygame.draw.rect(screen, green, [block[0], block[1], block_size, block_size])

def message(msg, color):
    mesg = font.render(msg, True, color)
    screen.blit(mesg, [width / 6, height / 3])

def gameLoop():
    game_over = False
    game_close = False

    x1 = width / 2
    y1 = height / 2

    x1_change = 0
    y1_change = 0

    snake_list = []
    length_of_snake = 1

    foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
    foody = round(random.randrange(0, height - block_size) / block_size) * block_size

    while not game_over:

        state = get_state([x1, y1], [foodx, foody])
        action = choose_action(state)

        if action == "LEFT":
            x1_change = -block_size
            y1_change = 0
        elif action == "RIGHT":
            x1_change = block_size
            y1_change = 0
        elif action == "UP":
            y1_change = -block_size
            x1_change = 0
        elif action == "DOWN":
            y1_change = block_size
            x1_change = 0

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        screen.fill(black)
        pygame.draw.rect(screen, red, [foodx, foody, block_size, block_size])
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for block in snake_list[:-1]:
            if block == snake_head:
                game_close = True

        draw_snake(block_size, snake_list)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
            foody = round(random.randrange(0, height - block_size) / block_size) * block_size
            length_of_snake += 1

        next_state = get_state([x1, y1], [foodx, foody])
        reward = get_reward([x1, y1], [foodx, foody], game_close)

        if state not in q_table:
            q_table[state] = {a: 0 for a in actions}
        if next_state not in q_table:
            q_table[next_state] = {a: 0 for a in actions}

        q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * max(q_table[next_state].values()) - q_table[state][action])

        clock.tick(15)

        if game_close:
            game_over = True

    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    pygame.quit()
    quit()

gameLoop()
