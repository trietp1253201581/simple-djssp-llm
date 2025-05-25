from problem import Problem, AVAIABLE_TERMINALS
from evaluate import Simulator
from model import CodeSegmentHDR
import random
random.seed(42)
problem = Problem(AVAIABLE_TERMINALS, pool_size=5)
problem.custom_generate(num_jobs=5, max_oprs_each_job=2,
                        num_machines=2, max_arr_time=10,
                        arrival_type='uniform', proc_dist='uniform', deadline_factor=1.4)

hdr = CodeSegmentHDR(code="""
def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    return mrel                   
""")

for job in problem.jobs:
    print(str(job))

simulator = Simulator(problem, hdr, hdr)

simulator.simulate(hdr, 0, True, 2)
