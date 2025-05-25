import copy
import random
from model import HDR
from problem import Problem
from typing import List
from abc import ABC
from evaluate import Evaluator

class Operator(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
    
    def __call__(self, *args, **kwds):
        pass
     
class Individual:
    DEFAULT_FITNESS: float = -1e8
    def __init__(self, problem: Problem):
        self.chromosome: HDR = None
        self.fitness = Individual.DEFAULT_FITNESS
        self.problem = problem
        
    def decode(self):
        return self.chromosome
        
    def cal_fitness(self, fitness_evaluator: Evaluator):
        sol = self.decode()
        if sol is None:
            self.fitness = Individual.DEFAULT_FITNESS
        else:
            self.fitness = fitness_evaluator([sol])[0][1]
            
            
class Population:
    def __init__(self, size: int, problem: Problem):
        self.size = size
        self.problem = problem
        self.inds: List[Individual] = []
        
    def cal_fitness(self, fitness_evaluator: Evaluator):
        hdrs = [ind.decode() for ind in self.inds]
        results = fitness_evaluator(hdrs)
        self.inds.clear()
        self.size = len(results)
        for i in range(self.size):
            new_ind = Individual(self.problem)
            new_ind.chromosome = results[i][0]
            new_ind.fitness = results[i][1]
            self.inds.append(new_ind)

class RandomSelectOperator(Operator):
    def __init__(self, problem):
        super().__init__(problem)
        
    def __call__(self, population: Population, sub_size: int) -> Population:
        if sub_size > population.size:
            return copy.deepcopy(population)
        
        sub_pop = Population(size=sub_size, problem=population.problem)
        sub_pop.inds = random.sample(population.inds, k=sub_size)
        return sub_pop
    
class TopKElitismReplaceOperator(Operator):
    def __init__(self, problem, k: int):
        super().__init__(problem)
        self.k = k
        
    def __call__(self, old_pop: Population, new_pop: Population, max_size: int) -> Population:
        sorted_inds = sorted(old_pop.inds, key=lambda x: x.fitness, reverse=True)
        elites = sorted_inds[:self.k]
        num_random = max_size - self.k
        remaining_inds = sorted_inds[self.k:] + new_pop.inds
        if num_random > 0:
            choosen = random.sample(remaining_inds, k=num_random)
        else:
            choosen = []
            
        inds = elites + choosen
        pop = Population(size=max_size, problem=old_pop.problem)
        pop.inds = inds
        return pop