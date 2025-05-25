import time
from model import CodeSegmentHDR, HDRException
from llm_support import LLM, LLMException
from basic_evo import Individual, Population, Operator
from evaluate import Evaluator
from abc import abstractmethod
import copy
import random
import logging
from typing import List, Dict

from problem import Problem

random.seed(42)

class MissingTemplateException(Exception):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
        
def get_template(template_file: str):
    try:
        with open(template_file, 'r') as f:
            lines = f.readlines()
            return "".join(lines)
    except FileNotFoundError:
        raise MissingTemplateException("Can't not load template function")
    
class LLMBaseOperator(Operator):
    def __init__(self, problem, llm_model: LLM, prompt_template: str):
        super().__init__(problem)
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self._logger = logging.getLogger(__name__)
        
    def _build_prompt(self, **config):
        return self.prompt_template.format(**config)
    
    @abstractmethod
    def _build_config(self, **kwargs) -> dict[str, str]:
        pass
    
    @abstractmethod
    def _process_json_response(self, data: dict):
        pass
    
    def _build_str_from_lst(self, data: list):
        return ", ".join(str(x) for x in data)
    
    def __call__(self, **kwargs):
        config = self._build_config(**kwargs)
        prompt = self._build_prompt(**config)
        response = self.llm_model.get_response(prompt)
        json_repsonse = self.llm_model.extract_response(response)
        return self._process_json_response(json_repsonse)
    
class LLMInitOperator(LLMBaseOperator):

    def _build_config(self, **kwargs):
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'init_size': str(kwargs.get('init_size')),
            'func_template': kwargs.get('func_template')
        }
    
    def _process_json_response(self, data):
        init_inds_code = data['init_inds']
        i = 0
        pop = Population(size=len(data['init_inds']), problem=self.problem)
        for code_json in init_inds_code:
            i += 1
            try:
                new_hdr = CodeSegmentHDR(code=code_json['code'])
                new_ind = Individual(self.problem)
                new_ind.chromosome = new_hdr
                pop.inds.append(new_ind)
            except HDRException as e:
                self._logger.error(str(type(e)) + e.msg, exc_info=True)
                continue
        return pop
    
    def __call__(self, init_size: int, func_template: str) -> Population:
        return super().__call__(init_size=init_size, func_template=func_template)
    
class LLMCrossoverOperator(LLMBaseOperator):
    def _build_config(self, **kwargs):
        p1: Individual = kwargs.get('p1')
        p2: Individual = kwargs.get('p2')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'hdr1': p1.chromosome.code,
            'hdr2': p2.chromosome.code,
        }
        
    def _process_json_response(self, data):
        recombined_code = data['recombined_hdr']
        inds: List[Individual] = []
        i = 0
        for code_json in recombined_code:
            try:
                i += 1
                new_hdr = CodeSegmentHDR(code=code_json['code'])
                new_ind = Individual(self.problem)
                new_ind.chromosome = new_hdr
                inds.append(new_ind)
            except HDRException as e:
                self._logger.error(str(e) + ":" + e.msg, exc_info=True)
                return None, None

        return inds[0], inds[1]
        
    def __call__(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        return super().__call__(p1=p1, p2=p2)
    
class LLMMutationOperator(LLMBaseOperator):
    
    def _build_config(self, **kwargs):
        p: Individual = kwargs.get('p')
        return {
            'job_str': self._build_str_from_lst(self.problem.jobs),
            'machine_str': self._build_str_from_lst(self.problem.machines),
            'terminal_set': self._build_str_from_lst(self.problem.terminals),
            'hdr': p.chromosome.code
        }
        
    def _process_json_response(self, data):
        try:
            code = data['rephrased_hdr']
            new_hdr = CodeSegmentHDR(code)
            new_ind = Individual(self.problem)
            new_ind.chromosome = new_hdr
        except HDRException as e:
            self._logger.error(str(type(e)) + ":" + e.msg, exc_info=True)
            return None
        return new_ind
    
    def __call__(self, p: Individual) -> Individual:
        return super().__call__(p=p)
    
class LLMEvoEngine:
    def __init__(self, problem: Problem, 
                 init_opr: Operator,
                 crossover_opr: Operator,
                 mutation_opr: Operator,
                 replacement_opr: Operator,
                 fitness_evaluator: Evaluator,
                 max_retries: int = 3):
        self.problem = problem
        self.init_opr = init_opr
        self.crossover_opr = crossover_opr
        self.mutation_opr = mutation_opr
        self.replacement_opr = replacement_opr
        self.fitness_evaluator = fitness_evaluator
        self.max_retries = max_retries
        
        self.gen = 0
        self.P: Population = None
        self.best: Individual = None
        self.solve_time: float = 0.0
        self._logger = logging.getLogger(__name__)
        
    def _retry(self, fn, *args, **kwargs):
        for attempt in range(1, self.max_retries+1):
            try:
                return fn(*args, **kwargs)
            except (LLMException, HDRException) as e:
                self._logger.warning(f"Attempt {attempt}/{self.max_retries} failed in {fn.__name__}: {e.msg}")
            except Exception as e:
                self._logger.error(f"Attempt {attempt}/{self.max_retries} failed in {fn.__name__}: {e}")
        raise Exception(f"All {self.max_retries} retries failed for {fn.__name__}")
    
    def _do_init(self, init_size, template):
        self._logger.info(f"Initializing population with {init_size} individuals")
        pop: Population = self.init_opr(init_size=init_size, func_template=template)
        self._logger.info(f"Population initialized with {len(pop.inds)} individuals")
        return pop
    
    def initialize(self, init_size: int, template: str) -> Population:
        return self._retry(self._do_init, init_size, template)

    def evaluate_pop(self, pop: Population):
        return self._retry(lambda: self._do_evaluate(pop))
    
    def _do_evaluate(self, pop: Population):
        pop.cal_fitness(self.fitness_evaluator)
        return pop
    
    def crossover_pop(self, parents: List[Individual], size: int, pc: float) -> List[Individual]:
        def _step():
            off = []
            while len(off) < size:
                if random.random() < pc:
                    p1, p2 = random.sample(parents, 2)
                    c1, c2 = self.crossover_opr(p1=p1, p2=p2)
                    if c1 is None: 
                        continue
                    off.extend([c1,c2])
            return off[:size]
        return self._retry(_step)
    
    def mutate(self, inds: List[Individual], pm: float) -> List[Individual]:
        def _step():
            out=[]
            cnt = 0
            for ind in inds:
                if random.random() < pm:
                    m = self.mutation_opr(p=ind)
                    if m is not None:
                        out.append(m)
                        cnt += 1
                    else:
                        out.append(copy.deepcopy(ind))
                else:
                    out.append(copy.deepcopy(ind))
            return out, cnt
        return self._retry(_step)
    
    def solve(self, max_gen: int, init_size: int, template_dir: str,
              pc: float = 0.8, pm: float = 0.1) -> Individual:
        self._logger.info("Starting evolutionary process")
        template = get_template(template_dir)
        start_time = time.time()
        
        # Init phase
        try:
            self.P = self.initialize(init_size, template)
            self.P = self.evaluate_pop(self.P)
        except Exception as e:
            self._logger.error(f"Initialization failed: {e}", exc_info=True)
            self.solve_time = time.time() - start_time
            return None
        
        while self.gen < max_gen:
            self._logger.info(f"Generation {self.gen}: Population size {len(self.P.inds)}")
            self.gen += 1
            
            # Crossover
            offsprings = self.crossover_pop(self.P.inds, self.P.size, pc)
            self._logger.info(f"Generated {len(offsprings)} offsprings from crossover")
            
            # Mutation
            mutated_offsprings, cnt = self.mutate(offsprings, pm)
            self._logger.info(f"Mutated {cnt} individuals out of {len(offsprings)} offsprings")
            
            # Evaluate new population
            self._logger.info("Evaluating new population")
            new_pop = Population(size=len(mutated_offsprings), problem=self.problem)
            new_pop.inds = mutated_offsprings
            new_pop = self.evaluate_pop(new_pop)
            self._logger.info(f"New population evaluated with {len(new_pop.inds)} individuals")
            
            # Replacement
            self.P = self.replacement_opr(self.P, new_pop, max_size=len(self.P.inds))
            self._logger.info(f"Population replaced, new size is {len(self.P.inds)}")
            
            # Update best individual
            if not self.best or max(ind.fitness for ind in self.P.inds) > self.best.fitness:
                self.best = max(self.P.inds, key=lambda ind: ind.fitness)
            
        self.solve_time = time.time() - start_time
        self._logger.info(f"Evolution completed in {self.solve_time:.2f} seconds")
        self._logger.info(f"Best individual found with fitness {self.best.fitness}")
        return self.best