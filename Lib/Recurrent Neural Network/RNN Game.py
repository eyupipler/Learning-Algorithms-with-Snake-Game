import pygame
import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers

pygame.init()

width, height = 640, 480
block_size = 20

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("RNN Snake Game")

clock = pygame.time.Clock()

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

font = pygame.font.SysFont(None, 35)

epsilon = 1.0  # Keşif oranı / Explore rate
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.95  # İndirim faktörü / Discount factor
alpha = 0.001  # Öğrenme oranı / Learning rate

memory = deque(maxlen=2000)
batch_size = 32
train_start = 1000

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

def create_rnn_model():
    model = tf.keras.models.Sequential([
        layers.LSTM(128, input_shape=(None, 2), return_sequences=True),
        layers.LSTM(128),
        layers.Dense(len(actions), activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')
    return model

model = create_rnn_model()

def get_state(snake_list, food):
    state = []
    for segment in snake_list:
        state.append([segment[0], segment[1]])
    state.append([food[0], food[1]])
    return np.array(state).reshape(1, -1, 2)

def choose_action(state):
    if np.random.rand() <= epsilon:
        return random.choice(actions)
    else:
        q_values = model.predict(state, verbose=0)
        return actions[np.argmax(q_values[0])]

def get_reward(snake_head, food, game_over):
    if game_over:
        return -100
    elif snake_head == food:
        return 100
    else:
        return -1

def replay():
    global epsilon
    if len(memory) < train_start:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma * np.amax(model.predict(next_state, verbose=0)[0]))
        target_f = model.predict(state, verbose=0)
        target_f[0][actions.index(action)] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def draw_snake(block_size, snake_list):
    for block in snake_list:
        pygame.draw.rect(screen, green, [block[0], block[1], block_size, block_size])

def message(msg, color):
    mesg = font.render(msg, True, color)
    screen.blit(mesg, [width / 6, height / 3])

def gameLoop():
    global epsilon

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        state = get_state(snake_list, [foodx, foody])
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

        next_state = get_state(snake_list, [foodx, foody])
        reward = get_reward([x1, y1], [foodx, foody], game_close)

        memory.append((state, action, reward, next_state, game_close))

        replay()

        clock.tick(15)

        if game_close:
            game_over = True

    pygame.quit()
    quit()

gameLoop()
