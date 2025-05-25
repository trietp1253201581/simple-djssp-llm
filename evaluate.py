from model import HDR, HDRException
import problem
from problem import Problem, Job, Machine, Operation, TerminalDictMaker, Terminal
from typing import List, Tuple
import logging
import copy
import time
from abc import ABC, abstractmethod
import pickle

class Simulator:
    DEFAULT_PRIOR: int = -1e9

    def __init__(self, problem: Problem, assign_job_default: HDR, assign_from_machine_pool_default: HDR):
        self.problem = problem
        self.waiting_pool: List[Job] = []
        self.pool_size = self.problem.pool_size
        self._logger = logging.getLogger(__name__)
        # Store default HDRs as class attributes
        self.assign_job_default = assign_job_default
        self.assign_from_machine_pool_default = assign_from_machine_pool_default

    def _print_with_debug(self, msg: str, debug: bool=False):
        if debug:
            print(msg)
            
    def _reset(self, jobs: List[Job], machines: List[Machine]):
        for machine in machines:
            machine.clear()
        self.waiting_pool.clear()

    def _calculate_priority(self, hdr: HDR, job: Job, machine: Machine, machines: List[Machine], curr_time: int):
        terminal_maker = TerminalDictMaker()
        terminal_maker.add_terminal(problem.JAT, job.time_arr)
        terminal_maker.add_terminal(problem.JCD, job.get_next_deadline())
        terminal_maker.add_terminal(problem.JD, job.get_job_deadline())
        terminal_maker.add_terminal(problem.JNPT, job.oprs[job.next_opr].get_avg_process_time())
        terminal_maker.add_terminal(problem.JRO, job.get_remain_opr())
        terminal_maker.add_terminal(problem.JRT, job.get_remain_process_time())
        terminal_maker.add_terminal(problem.JTPT, job.get_total_process_time())
        terminal_maker.add_terminal(problem.JS, job.get_slack_time(curr_time))
        terminal_maker.add_terminal(problem.JW, job.weight)
        terminal_maker.add_terminal(problem.JWT, job.wait_time)
        terminal_maker.add_terminal(problem.ML, machine.get_remain_time(curr_time))
        terminal_maker.add_terminal(problem.MR, machine.get_remain_time(curr_time))
        terminal_maker.add_terminal(problem.MREL, machine.get_relax_time(curr_time))
        terminal_maker.add_terminal(problem.MPR, machine.processed_count)
        terminal_maker.add_terminal(problem.MUTIL, machine.get_util(curr_time))
        terminal_maker.add_terminal(problem.TNOW, curr_time)
        util_system = sum(m.get_util(curr_time) for m in machines) / len(machines) if machines else 0
        terminal_maker.add_terminal(problem.UTIL, util_system)
        avg_wait = sum(j.wait_time for j in self.waiting_pool) / len(self.waiting_pool) if self.waiting_pool else 0
        terminal_maker.add_terminal(problem.AVGWT, avg_wait)

        try:
            score = hdr.execute(**terminal_maker.var_dicts)
            self._print_with_debug(f'\t\t\tCalculate priority for job {job.id}, machine {machine.id}: {score}')
            return score
        except HDRException as e:
            self._logger.warning(f"HDR Exception: {e.msg}, use DEFAULT PRIOR instead", exc_info=True)
            return Simulator.DEFAULT_PRIOR

    def simulate(self, hdr: HDR, apply_type: int=0, debug: bool = False, sleep_time: int | None = None):
        jobs = copy.deepcopy(self.problem.jobs)
        machines = copy.deepcopy(self.problem.machines)

        total_jobs = len(jobs)
        completed_jobs = 0
        curr_time = 0

        self._reset(jobs, machines)

        while completed_jobs < total_jobs:
            self._print_with_debug(f"Current time: {curr_time}---------", debug)

            # Check for new jobs arriving at the current time
            for job in jobs:
                if job.time_arr <= curr_time and job.status == Job.Status.ARRIVED:
                    self.waiting_pool.append(job)
                    job.wait_time = 0
                    job.status = Job.Status.WAITING
                    self._print_with_debug(f"\tJob {job.id} arrived and added to waiting pool", debug)
            pool_str = f'[{", ".join(str(job.id) for job in self.waiting_pool)}]'
            self._print_with_debug(f"\tWaiting pool: {pool_str}", debug)
            
            # Calculate priorities to choose machines which can process jobs
            apply_hdr = self.assign_job_default if apply_type == 2 else hdr
            for job in self.waiting_pool[:]:
                next_opr = job.oprs[job.next_opr]
                best_machine: Machine = None
                best_priority = Simulator.DEFAULT_PRIOR
                available_machines_id = [m.id for m in next_opr.available_machines.keys()]
                for machine in machines:
                    if machine.id not in available_machines_id:
                        continue
                    prior = self._calculate_priority(apply_hdr, job, machine, machines, curr_time)
                    if prior > best_priority:
                        best_priority = prior
                        best_machine = machine
                if best_machine is not None:
                    job.status = Job.Status.READY
                    best_machine.assign(job)
                    self._print_with_debug(f"\tJob {job.id} assigned to machine {best_machine.id} with priority {best_priority}", debug)
                    self.waiting_pool.remove(job)
            self._print_with_debug(f"\tWaiting pool after assignment: {pool_str}", debug)
            
            # Print the each machine's pool
            for machine in machines:
                pool_jobs = ', '.join(f"J{job.id}.Op{job.next_opr}" for job in machine.pool)
                self._print_with_debug(f"\t\tMachine {machine.id}: {machine.get_status()}, pool=[{pool_jobs}]", debug)
            
            # Check if any machine completes its now processing job
            for machine in machines:
                if machine.get_status() == Machine.Status.RELAX:
                    continue
                if curr_time >= machine.finish_time:
                    job = machine.curr_job
                    job.next_opr += 1
                    if job.next_opr == len(job.oprs):
                        job.status = Job.Status.FINISHED
                        completed_jobs += 1
                    else:
                        job.status = Job.Status.WAITING
                        job.wait_time = 0
                        self.waiting_pool.append(job)
                    job.finish_time = curr_time
                    machine.processed_count += 1
                    machine.curr_job = None 
                    self._print_with_debug(f"\tMachine {machine.id} completed operation {job.next_opr - 1} of job {job.id}.", debug)
                    
            # Check if any machine is idle and can take a job from the its private pool
            apply_hdr = self.assign_from_machine_pool_default if apply_type == 1 else hdr
            for machine in machines:
                if machine.get_status() == Machine.Status.PROCESSING:
                    continue
                if not machine.pool:
                    continue
                best_job: Job = None
                best_priority = Simulator.DEFAULT_PRIOR
                for job in machine.pool:
                    prior = self._calculate_priority(apply_hdr, job, machine, machines, curr_time)
                    if prior > best_priority:
                        best_priority = prior
                        best_job = job
                if best_job is not None:
                    best_job.status = Job.Status.PROCESSING
                    self._print_with_debug(f"\tMachine {machine.id} assigned job {best_job.id} from its pool with priority {best_priority}", debug)
                    machine.pool.remove(best_job)
                    machine.curr_job = best_job
                    machine.finish_time = curr_time + best_job.oprs[best_job.next_opr].available_machines[machine]
                    self._print_with_debug(f"\t\tMachine {machine.id} will finish job {best_job.id} at time {machine.finish_time}", debug)

            curr_time += 1
            if debug:
                time.sleep(0.02 if sleep_time is None else sleep_time)

        makespan = max(m.finish_time for m in machines)
        self._print_with_debug(f"Done!, makespan = {makespan}", debug)
        return makespan
    
class Evaluator(ABC):
    def __init__(self, problem: Problem):
        self.problem = problem
        
    @abstractmethod
    def __call__(self, hdrs: List[HDR]) -> List[Tuple[HDR, float]]:
        pass
    
    def save_state(self, checkpoint_path: str, fields_to_save: list|None = None):
        with open(checkpoint_path, 'wb') as f:
            if fields_to_save is None:
                pickle.dump(self, f)
            else:
                data = {field: getattr(self, field) for field in fields_to_save if hasattr(self, field)}
                pickle.dump(data, f)
            
    def load_state(self, checkpoint_path: str, fields_to_update: list | None = None):
        with open(checkpoint_path, 'rb') as f:
            loaded = pickle.load(f)
    
            if isinstance(loaded, dict):
                # Nếu file chứa dict, thì lấy từ dict
                if fields_to_update is None:
                    for field, value in loaded.items():
                        setattr(self, field, value)
                else:
                    for field in fields_to_update:
                        if field in loaded:
                            setattr(self, field, loaded[field])
            else:
                # Nếu file chứa nguyên object
                if fields_to_update is None:
                    self.__dict__.update(loaded.__dict__)
                else:
                    for field in fields_to_update:
                        if hasattr(loaded, field):
                            setattr(self, field, getattr(loaded, field))
    
class SimulationBaseEvaluator(Evaluator):
    def __init__(self, problem):
        super().__init__(problem)
        self.simulator = Simulator(self.problem)
        self._logger = logging.getLogger(__name__)
        
    def __call__(self, hdrs) -> List[Tuple[HDR, float]]:
        self._logger.info(f'Start evaluate {len(hdrs)} HDR.')
        results = []
        for id, hdr in enumerate(hdrs):
            self._logger.info(f'Evaluate HDR {id + 1}/{len(hdrs)}')
            makespan = self.simulator.simulate(hdr, debug=False)
            fitness = -makespan
            results.append((hdr, fitness))
        self._logger.info(f'Successfully evaluate {len(results)}/{len(hdrs)} HDR.')
        return results