import pygame
import random
import numpy as np

pygame.init()

width, height = 640, 480
block_size = 20

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("GA Snake Game")

clock = pygame.time.Clock()

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

font = pygame.font.SysFont(None, 35)

# GA
population_size = 100
mutation_rate = 0.01
num_generations = 50
chromosome_length = 100  # Hareket sayısı / Number of moves

actions = ["UP", "DOWN", "LEFT", "RIGHT"]


def create_population():
    return [random.choices(actions, k=chromosome_length) for _ in range(population_size)]


def fitness_function(chromosome):
    x1, y1 = width // 2, height // 2
    snake_list = []
    length_of_snake = 1
    foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
    foody = round(random.randrange(0, height - block_size) / block_size) * block_size

    score = 0
    for action in chromosome:
        if action == "LEFT":
            x1 -= block_size
        elif action == "RIGHT":
            x1 += block_size
        elif action == "UP":
            y1 -= block_size
        elif action == "DOWN":
            y1 += block_size

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0 or [x1, y1] in snake_list:
            return score

        if x1 == foodx and y1 == foody:
            score += 1
            foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
            foody = round(random.randrange(0, height - block_size) / block_size) * block_size
            length_of_snake += 1

        snake_head = [x1, y1]
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

    return score


def crossover(parent1, parent2):
    crossover_point = random.randint(0, chromosome_length - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(chromosome):
    for i in range(chromosome_length):
        if random.random() < mutation_rate:
            chromosome[i] = random.choice(actions)
    return chromosome


def select_parents(population, fitnesses):
    fitnesses = np.array(fitnesses, dtype=np.float64)
    fitnesses[np.isnan(fitnesses)] = 0
    selected = np.random.choice(range(population_size), size=2, p=fitnesses / fitnesses.sum())
    return population[selected[0]], population[selected[1]]


def evolve_population(population):
    fitnesses = [fitness_function(individual) for individual in population]
    new_population = []
    for _ in range(population_size // 2):
        parent1, parent2 = select_parents(population, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([mutate(child1), mutate(child2)])
    return new_population


def draw_snake(block_size, snake_list):
    for block in snake_list:
        pygame.draw.rect(screen, green, [block[0], block[1], block_size, block_size])


def message(msg, color):
    mesg = font.render(msg, True, color)
    screen.blit(mesg, [width / 6, height / 3])


def gameLoop():
    population = create_population()

    for generation in range(num_generations):
        print(f"Generation {generation + 1}")
        population = evolve_population(population)

    best_individual = max(population, key=fitness_function)
    x1 = width // 2
    y1 = height // 2
    snake_list = []
    length_of_snake = 1
    foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
    foody = round(random.randrange(0, height - block_size) / block_size) * block_size

    for action in best_individual:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if action == "LEFT":
            x1 -= block_size
        elif action == "RIGHT":
            x1 += block_size
        elif action == "UP":
            y1 -= block_size
        elif action == "DOWN":
            y1 += block_size

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
            break

        screen.fill(black)
        pygame.draw.rect(screen, red, [foodx, foody, block_size, block_size])
        snake_head = [x1, y1]
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for block in snake_list[:-1]:
            if block == snake_head:
                message("You Lost!", red)
                pygame.display.update()
                pygame.time.delay(2000)
                pygame.quit()
                quit()

        draw_snake(block_size, snake_list)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - block_size) / block_size) * block_size
            foody = round(random.randrange(0, height - block_size) / block_size) * block_size
            length_of_snake += 1

        clock.tick(15)


gameLoop()
