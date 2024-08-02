import pygame
import numpy as np
import random
import math

pygame.init()

width, height = 640, 480
block_size = 20

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("CSA Snake Game")

clock = pygame.time.Clock()

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

font = pygame.font.SysFont(None, 35)

# CSA
num_crows = 5
num_dimensions = 2  # x ve y koordinatları / x and y coordinates
num_iterations = 100
flight_length = 2
awareness_probability = 0.1

crows_positions = np.random.randint(low=0, high=width//block_size, size=(num_crows, num_dimensions)) * block_size
crows_memory = np.copy(crows_positions)

snake_list = []
length_of_snake = 1
x1 = width // 2
y1 = height // 2
foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
foody = round(random.randrange(0, height - block_size) / block_size) * block_size

def distance(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2)**2))

def update_positions(crows_positions, crows_memory):
    for i in range(num_crows):
        if random.random() > awareness_probability:
            j = random.randint(0, num_crows - 1)
            crows_positions[i] = crows_positions[i] + flight_length * (crows_memory[j] - crows_positions[i])
        else:
            crows_positions[i] = np.random.randint(low=0, high=width//block_size, size=num_dimensions) * block_size
        crows_positions[i] = np.clip(crows_positions[i], 0, width - block_size)
    return crows_positions

def evaluate_positions(crows_positions, food_position):
    distances = np.array([distance(crow, food_position) for crow in crows_positions])
    best_index = np.argmin(distances)
    return crows_positions[best_index], distances[best_index]

def draw_snake(block_size, snake_list):
    for block in snake_list:
        pygame.draw.rect(screen, green, [block[0], block[1], block_size, block_size])

def message(msg, color):
    mesg = font.render(msg, True, color)
    screen.blit(mesg, [width / 6, height / 3])

def gameLoop():
    global snake_list, length_of_snake, x1, y1, foodx, foody, crows_positions, crows_memory

    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        crows_positions = update_positions(crows_positions, crows_memory)
        best_position, best_distance = evaluate_positions(crows_positions, np.array([foodx, foody]))

        x1, y1 = best_position
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
                game_over = True

        draw_snake(block_size, snake_list)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
            foody = round(random.randrange(0, height - block_size) / block_size) * block_size
            length_of_snake += 1

        if best_distance == 0:
            crows_memory = np.copy(crows_positions)

        clock.tick(15)

    pygame.quit()
    quit()

gameLoop()
