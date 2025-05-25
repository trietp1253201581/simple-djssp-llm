from enum import Enum
from typing import List
import random

from typing import Dict

class Terminal:
    """
    Terminal is a variable that can be used in the heuristic dispatching rule (HDR)
    """
    def __init__(self, label: str, description: str=""):
        self.label = label
        self.description = description
        
    def __str__(self):
        return f'{self.label}({self.description})'
    
    def __hash__(self):
        return hash(self.label)
    
    def __eq__(self, value):
        if not isinstance(value, Terminal):
            return False
        return self.label == value.label
    
JNPT = Terminal('jnpt', "Next operation processing time of a job")
JTPT = Terminal('japt', "Total process time of job")
JRT = Terminal('jrt', "Remaining time to complete job")
JRO = Terminal('jro', "Remaining operations to complete job")
JWT = Terminal('jwt', "Job waiting time in waiting pool and job pool")
JAT = Terminal('jat', "Arriving time of a job")
JD = Terminal('jd', "Job Deadline")
JCD = Terminal('jcd', "Deadline of next operation of a job")
JS = Terminal('js', "Slack time of a job")
JW = Terminal('jw', "Weight of job")
ML = Terminal('ml', "Time from process current job")
MR = Terminal('mr', "Remaining time to completed current job")
MREL = Terminal('mrel', "Relaxing Time of machine")
MPR = Terminal('mpr', "Num of processed opr in this machine")
MUTIL = Terminal('mutil', "Now utilization")
TNOW = Terminal('tnow', "Current time of system")
UTIL = Terminal('util', "Now utilization of system")
AVGWT = Terminal('avgwt', "Average wait time of all jobs")

AVAIABLE_TERMINALS = [
    JNPT, JAT, JCD, JD, JRO, JRT, JS, JW, JWT, JTPT,
    ML, MR, MREL, MPR, MUTIL, 
    TNOW, UTIL, AVGWT
]

class TerminalDictMaker:
    """
    Make a dictionary of terminals and their values
    """
    def __init__(self):
        self.var_dicts: Dict[str, any] = {}
        
    def add_terminal(self, new_terminal: Terminal, new_value: int|float):
        self.var_dicts[new_terminal.label] = new_value
        
class Operation:
    """
    Operation is a job's operation
    """
    def __init__(self, deadline: float, 
                 available_machines: dict['Machine', float]):
        
        self.deadline = deadline
        self.available_machines = available_machines
        self.avg_process_time = None

    def get_avg_process_time(self):
        """
        Get the average process time of the operation
        """
        if self.avg_process_time is not None:
            return self.avg_process_time
        total_process_time = 0.0
        for _, p in self.available_machines.items():
            total_process_time += p
        return total_process_time/len(self.available_machines)
    
    def __str__(self):
        available_m_str = ""
        for machine, process_time in self.available_machines.items():
            available_m_str += str(machine.id) + ":" + str(f'{process_time:.1f}') + ", "
        return f'Operation(deadline={self.deadline}, available_machines=[{available_m_str}])'
    
class Job:
    class Status(Enum):
        ARRIVED = 1
        WAITING = 2
        READY = 3
        PROCESSING = 0
        FINISHED = -1
        
    def __init__(self, id: int):
        self.id = id
        self.next_opr = 0
        self.time_arr = 0
        self.finish_time = 0
        self.wait_time = 0
        self.weight = 1.0
        self.oprs: List[Operation] = []
        self.status = Job.Status.ARRIVED
        self.prior = 0
        
    def get_remain_opr(self):
        """
        Get the number of remaining operations of the job
        """
        return len(self.oprs) - self.next_opr
    
    def get_remain_process_time(self):
        """
        Get the remaining process time of the job
        """
        remain_process_time = 0
        for i in range(self.next_opr, len(self.oprs)):
            remain_process_time += self.oprs[i].get_avg_process_time()
        return remain_process_time
    
    def get_total_process_time(self):
        """
        Get the total process time of the job
        """
        total_process_time = 0
        for opr in self.oprs:
            total_process_time += opr.get_avg_process_time()
        return total_process_time
    
    def get_next_deadline(self):
        """
        Get the deadline of the next operation of the job
        """
        return self.oprs[self.next_opr].deadline
    
    def get_job_deadline(self):
        """
        Get the deadline of the job
        """
        return self.oprs[-1].deadline
    
    def get_slack_time(self, curr_time: float):
        """
        Get the slack time of the job
        """
        return max(0, self.get_job_deadline() - curr_time - self.get_remain_process_time())
        
    def add_opr(self, new_opr: Operation):
        """
        Add a new operation to the job
        """
        self.oprs.append(new_opr)
    
    def __str__(self):
        return f'Job(id={self.id}, status={self.status}, time_arr={self.time_arr}, next_opr={self.next_opr}, oprs=[{", ".join(str(x) for x in self.oprs)}])'
    
    def __eq__(self, value):
        if not isinstance(value, Job):
            return False
        return self.id == value.id
    
    def __hash__(self):
        return hash(self.id)
        
        
class Machine:
    class Status(Enum):
        RELAX = 0
        PROCESSING = 1
        
    def __init__(self, id: int):
        self.id = id
        self.curr_job: Job = None
        self.pool: list[Job] = []
        self.finish_time = 0
        self.processed_count = 0
        self.processed_time = 0
        
    def clear(self):
        """
        Clear the machine
        """
        self.curr_job = None
        self.finish_time = 0
        self.processed_count = 0
        self.pool.clear()
        
    def assign(self, new_job: Job):
        self.pool.append(new_job)
        
    def get_status(self) -> Status:
        """
        Get the status of the machine
        """
        if self.curr_job is None:
            return Machine.Status.RELAX
        return Machine.Status.PROCESSING
    
    def get_relax_time(self, curr_time: float):
        """
        Get the relax time of the machine
        """
        return max(0, curr_time - self.finish_time)
    
    def get_remain_time(self, curr_time: float):
        """
        Get the remain time of the machine
        """
        return max(0, self.finish_time - curr_time)
    
    def get_util(self, curr_time: float):
        """
        Get the utilization of the machine
        """
        return self.processed_time / curr_time if curr_time > 0 else 0
    
    def get_remain_job(self):
        return len(self.pool)
        
    def __str__(self):
        return f'Machine(id={self.id}, curr={self.curr_job.id if self.curr_job is not None else None}, finish_time={self.finish_time}, processed={self.processed_count}])'
    
    def __eq__(self, value):
        if not isinstance(value, Machine):
            return False
        return self.id == value.id
    
    def __hash__(self):
        return hash(self.id)

class Problem:
    def __init__(self, terminals: List[Terminal], pool_size: int):
        """
        Initialize the DJSSP problem

        Args:
            terminals (List[Terminal]): List of terminals
            pool_size (int): Size of job pool
        """
        self.jobs: List[Job] = []
        self.machines: List[Machine] = []
        self.terminals = terminals
        self.pool_size = pool_size
        
    def random_generate(self, num_jobs: int, max_oprs_each_job: int, num_machines: int, max_arr_time = 1000):
        """
        Total Randomly generate a DJSSP problem

        Args:
            num_jobs (int): Number of jobs
            max_oprs_each_job (int): Maximum number of operations for each job
            num_machines (int): Number of machines
            max_arr_time (int): Maximum arrival time
        """
        self.jobs = []
        self.machines = [Machine(i) for i in range(num_machines)]
        for i in range(num_jobs):
            new_job = Job(id=i)
            new_job.time_arr = random.randint(0, max_arr_time)
            new_job.prior = random.randint(1, 5)
            num_opr = random.randint(1, max_oprs_each_job)
            for pos in range(num_opr):
                last_d = new_job.oprs[-1].deadline if len(new_job.oprs) > 0 else 0
                d = random.randint(last_d + 1, max_arr_time * 3 // 2 + last_d)
                K_ids = random.sample(range(num_machines), random.randint(1, num_machines))
                K = [self.machines[id] for id in K_ids]
                available_machines = dict()
                for m in K:
                    p = random.randint(1, max_arr_time * 4 // 3)
                    available_machines[m] = p
                new_opr = Operation(d, available_machines)
                new_job.add_opr(new_opr)
            self.jobs.append(new_job)
            
    def custom_generate(self, num_jobs:int, max_oprs_each_job:int, num_machines:int, max_arr_time:int = 1000,
                        arrival_type: str='uniform', proc_dist: str='uniform', deadline_factor: float=2.0):
        """
        Custom generate a DJSSP problem

        Args:
            num_jobs (int): Number of jobs
            max_oprs_each_job (int): Maximum number of operations for each job
            num_machines (int): Number of machines
            max_arr_time (int): Maximum arrival time
            arrival_type (str): Type of arrival time
            proc_dist (str): Type of processing time
            deadline_factor (float): Factor of deadline
        """
        self.jobs = []
        self.machines = [Machine(i) for i in range(num_machines)]
        
        for j in range(num_jobs):
            new_job = Job(id=j)
            
            if arrival_type == 'uniform':
                new_job.time_arr = int(random.uniform(0, max_arr_time))
            elif arrival_type == 'gauss':
                new_job.time_arr = int(random.gauss(max_arr_time / 2, max_arr_time / 6))
            elif arrival_type == 'burst':
                if random.random() < 0.8:
                    new_job.time_arr = int(random.uniform(0, max_arr_time / 4))
                else:
                    new_job.time_arr = int(random.uniform(3 * max_arr_time / 4, max_arr_time))
            else:
                new_job.time_arr = random.randint(0, max_arr_time)
            
            new_job.prior = random.randint(1, 5)
            num_opr = random.randint(1, max_oprs_each_job)
            
            oprs = []
            total_duration = 0
            for _ in range(num_opr):
                if proc_dist == 'uniform':
                    duration = int(random.uniform(1, 2 * max_arr_time))
                elif proc_dist == 'bimodal':
                    duration = random.choice([int(random.uniform(1, max_arr_time/3)), int(random.uniform(2 * max_arr_time/3, 2 * max_arr_time))])
                else:
                    duration = int(random.gauss(max_arr_time / 2, max_arr_time / 6))
                
                total_duration += duration
                
                selected_machine_ids = random.sample(range(num_machines), k=2)
                available_machines = {
                    self.machines[m_id]: duration + random.randint(-max_arr_time//5, max_arr_time//5) for m_id in selected_machine_ids
                }
                
                # Deadline tính dồn, có thể cộng thêm thời gian ngẫu nhiên hoặc cố định
                last_deadline = oprs[-1].deadline if oprs else new_job.time_arr
                deadline = last_deadline + deadline_factor * duration
                
                oprs.append(Operation(deadline, available_machines))
                
            new_job.oprs = oprs
            self.jobs.append(new_job)