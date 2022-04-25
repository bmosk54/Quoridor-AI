from src.player.geneticBot import *
from src.Game import *
import tensorflow
import numpy
import pygad

class genetic():

    POPULATION_SIZE = 4
    FEATURE_NUM = 4
    HASFEAT_PROB = 0.5
    fitness = numpy.zeros(POPULATION_SIZE)
    population = numpy.zeros((POPULATION_SIZE, FEATURE_NUM))

    def create_population(self):
        for i in range(self.POPULATION_SIZE):
            for j in range(self.FEATURE_NUM):
                if(random.uniform(0,1) <= self.HASFEAT_PROB):
                    self.population[i][j] = random.uniform(-1,1)
                else:
                    self.population[i][j] = 0

    def reset_fitness(self):
        self.fitness = [0] * self.POPULATION_SIZE

    def play(self):
        self.reset_fitness()
        AI_1 = geneticBot('1')
        AI_2 = geneticBot('2')
        players = [AI_1, AI_2]
        for i in range(self.POPULATION_SIZE):
            for j in range(self.POPULATION_SIZE):
                if(i != j ):
                    AI_1.setChromosome(self.population[i])
                    AI_2.setChromosome(self.population[j])
                    game = Game(players)
                    game.start(2)
                    if (game.getGameEnded(AI_1) == 1 ):
                        self.fitness[i] +=1
                    if (game.getGameEnded(AI_2) == 1 ):
                        self.fitness[j] +=1
                    game.end()


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

input_layer  = tensorflow.keras.layers.Input(4)
dense_layer1 = tensorflow.keras.layers.Dense(5, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(1, activation="linear")(dense_layer1)

model = tensorflow.keras.Model(inputs=input_layer,
                               outputs=output_layer)

weights_vector = pygad.kerasga.model_weights_as_vector(model=model)
keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=4)

g = genetic()
data_inputs = g.create_population()
g.play()
data_outputs = g.fitness

num_generations = 20
num_parents_mating = 2
initial_population = keras_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=g.fitness,
                       on_generation=callback_generation)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))