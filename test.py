import os
from problem import Problem, AVAIABLE_TERMINALS
from evaluate import Simulator, SimulationBaseEvaluator
from model import CodeSegmentHDR
from llm_support import GoogleAIStudioLLM
import prompt_template as pt 
from basic_evo import TopKElitismReplaceOperator
from llm_evo import LLMInitOperator, LLMCrossoverOperator, LLMMutationOperator, LLMEvoEngine
import random
from datetime import datetime
# Set logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Định dạng chung
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(name)s (%(filename)s:%(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler ghi vào file
file_handler = logging.FileHandler(f'process_{datetime.now().strftime("%Y_%m_%d")}.log')
file_handler.setFormatter(formatter)

# Handler ghi ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Thêm cả 2 handler vào logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

random.seed(42)
problem = Problem(AVAIABLE_TERMINALS, pool_size=5)
problem.custom_generate(num_jobs=15, max_oprs_each_job=3,
                        num_machines=4, max_arr_time=20,
                        arrival_type='uniform', proc_dist='uniform', deadline_factor=1.4)

llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='config/llm_core.json',
                              runtime_config='config/llm_runtime.json')

init_opr = LLMInitOperator(problem, llm_model, pt.INIT_IND_PROMPT_TEMPLATE)
crossover_opr = LLMCrossoverOperator(problem, llm_model, pt.CROSSOVER_PROMPT_TEMPLATE)
mutation_opr = LLMMutationOperator(problem, llm_model, pt.MUTATION_PROMPT_TEMPLATE)
replace_opr = TopKElitismReplaceOperator(problem, k=2)

hdr1 = CodeSegmentHDR(code="""
def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    return -jnpt                   
""")

hdr2 = CodeSegmentHDR(code="""
def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    return mrel
""")

evaluator = SimulationBaseEvaluator(Simulator(problem, hdr1, hdr2))

engine = LLMEvoEngine(problem, init_opr, crossover_opr, mutation_opr, replace_opr,
                      evaluator, max_retries=3)

best = engine.solve(max_gen=10, init_size=12, template_dir='template.txt')

if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
    print(f"Time: {engine.solve_time:.2f}s")
    os.makedirs('best_solution', exist_ok=True)
    best.chromosome.save(f'best_after_gen_{engine.gen}.py')
llm_model.close()


