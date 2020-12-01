import json
import logging
from copy import deepcopy
from random import random
from time import time, perf_counter
from typing import Union, Tuple, List, Dict, Optional

from data_objects import Location, data_globals
from hgsadc import HGSADC
from individual import Individual, Route
from model import run_settings
from population import Population
from settings import Data
from util import create_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s: %(message)s")
logger = create_logger(__name__)


class Runner:
    """Creates and runs HGSADC instances."""
    RUN = 1
    DECOMPOSE = 2
    STOP = 3

    def __init__(self, max_non_improving_iterations: int, max_run_time: float, use_multiprocessing: bool = False,
                 possible_locations=None, possible_vehicle_types=None, decomposition: bool = False,
                 seeded_solutions: List[Individual] = None):
        if possible_vehicle_types is None:
            possible_vehicle_types = data_globals.ALL_VEHICLE_TYPES
        if possible_locations is None:
            possible_locations = data_globals.ALL_CUSTOMERS
        self.max_non_improving_iterations = max_non_improving_iterations
        self.max_run_time = max_run_time
        self.use_multiprocessing = use_multiprocessing
        self.possible_locations = possible_locations
        self.possible_vehicle_types = possible_vehicle_types
        self.decomposition = decomposition
        self.seeded_solutions = seeded_solutions

        logger.debug(f"Possible locations: {possible_locations}\nPossible vehicle types: {possible_vehicle_types}")

    def run(self) -> Union[Optional[Individual], List[Optional[Individual]]]:
        """Runs the appropriate run function, according to whether multiprocessing is enabled"""

        if self.use_multiprocessing:
            # Create process manager to synchronise values between processes
            # with multiprocessing.Manager() as sync_manager:
            #     # Create populations with process manager
            #     feasible_population = Population(process_manager=sync_manager)
            #     infeasible_population = Population(process_manager=sync_manager)
            #     # Create HGSADC object
            #     algorithm = HGSADC(feasible_population=feasible_population,
            #                        infeasible_population=infeasible_population,
            #                        possible_locations=self.possible_locations,
            #                        possible_vehicle_types=self.possible_vehicle_types)
            #
            #     return self._run_with_multiprocessing(sync_manager, algorithm,
            #                         run_settings.RUN_CONFIG.available_cores * run_settings.RUN_CONFIG.core_proportion)
            pass
        else:
            feasible_population = Population()
            infeasible_population = Population()
            # Create HGSADC object
            algorithm = HGSADC(feasible_population=feasible_population, infeasible_population=infeasible_population,
                               possible_locations=self.possible_locations,
                               possible_vehicle_types=self.possible_vehicle_types)

            return self._run_with_single_processing(algorithm)

    def _run_with_single_processing(self, algorithm: HGSADC) -> Union[Optional[Individual], List[Optional[Individual]]]:
        """Run the algorithm and return the best feasible solution.

        This method runs the algorithm in the manner depicted in Algorithm 1 of Vidal et al. (2012).
        Numbered comments indicate correlated lines from Vidal's Algorithm 1."""
        logger.info("Initialising single process run.")
        # Initialise variables
        iterations = 0
        best_iteration = 0
        non_improving_iterations = 0
        capacity_feasible_count = 0
        duration_feasible_count = 0
        best_feasible_solution = None

        if self.seeded_solutions:
            logger.info(f"Seeding population with {len(self.seeded_solutions)} solutions.")
            for solution in self.seeded_solutions:
                algorithm.add_to_population(solution)
                if algorithm.compare_solution_to_best(best_feasible_solution, solution):
                    best_feasible_solution = solution
                    logger.info(f"Seeded solution {solution} set as best:\n{json.dumps(solution.routes_to_dict())}")
                else:
                    logger.info(f"{solution}\n{json.dumps(solution.routes_to_dict())}")

        run_start_time = time()

        # 1: Initialize population
        best_feasible_solution = algorithm.generate_random_individuals(4 * run_settings.RUN_CONFIG.min_pop_size,
                                                                       best_feasible_solution)
        solutions_generated = 4 * run_settings.RUN_CONFIG.min_pop_size
        # if self.decomposition:
        best_list: List[Optional[Individual]] = []
        logger.info(f"Starting best solution: {best_feasible_solution}")

        logger.info("Beginning main iteration loop")
        # 2: Keep going while the max non-improving iterations and max run time have not been reached
        while non_improving_iterations < self.max_non_improving_iterations and (
                self.max_run_time < 0 or time() - run_start_time < self.max_run_time):
            # logger.debug(f"Iteration {iterations + 1}")
            # 3, 4, 5 & 6: Create, educate and repair a new individual
            child = self._create_and_educate_individual(algorithm)
            # Increment the total number of iterations
            iterations += 1
            solutions_generated += 1
            # 6 & 7: Insert the new solution into the appropriate population
            # new_best_child = algorithm.add_to_population(child, best_feasible_solution)
            algorithm.add_to_population(child)
            if not child.feasible:
                child = deepcopy(child)
                child.educate(10)
                if not child.feasible:
                    child.educate(100)

                if child.feasible:
                    algorithm.add_to_population(child)

            if algorithm.compare_solution_to_best(best_feasible_solution, child):
                # If this is a decomposition, keep the best three solutions
                if self.decomposition and best_feasible_solution is not None:
                    best_list.append(best_feasible_solution)
                    if len(best_list) > 2:
                        best_list.pop(0)

                # If the new solution improves on the best feasible, then record it and reset non-improving iterations
                best_feasible_solution = child
                non_improving_iterations = 0
                best_iteration = iterations
                logger.info(f"New best solution: \n{child}")
            else:
                non_improving_iterations += 1

            # 8: Check whether either population has reached the maximum size
            if len(algorithm.feasible_population.population) >= \
                    run_settings.RUN_CONFIG.min_pop_size + run_settings.RUN_CONFIG.generation_size:
                algorithm.feasible_population.cull_population(run_settings.RUN_CONFIG.min_pop_size)
            if len(algorithm.infeasible_population.population) >= \
                    run_settings.RUN_CONFIG.min_pop_size + run_settings.RUN_CONFIG.generation_size:
                algorithm.infeasible_population.cull_population(run_settings.RUN_CONFIG.min_pop_size)

            # 9: If the best solution hasn't been improved for too long, then diversify the population
            if (non_improving_iterations + 1) % (
                    run_settings.RUN_CONFIG.diversification_proportion * self.max_non_improving_iterations) == 0:
                logger.info(f"Diversifying on iteration {iterations}")
                algorithm.diversify_populations(best_feasible_solution)
                solutions_generated += 4 * run_settings.RUN_CONFIG.min_pop_size

            # Count the number of individuals that are naturally feasible with respect to each type of penalty
            if child.capacity_penalty == 0:
                capacity_feasible_count += 1
            if child.duration_penalty == 0:
                duration_feasible_count += 1
            # 10: Adjust the penalty parameters after a certain number of iterations
            if iterations % run_settings.RUN_CONFIG.parameter_adjustment_frequency == 0:
                logger.info(f"Adjusting parameters on iteration {iterations}")
                run_settings.RUN_CONFIG.adjust_penalties(
                    capacity_feasible_count / run_settings.RUN_CONFIG.parameter_adjustment_frequency,
                    duration_feasible_count / run_settings.RUN_CONFIG.parameter_adjustment_frequency)
                capacity_feasible_count = 0
                duration_feasible_count = 0

            # 11: Decompose the problem if the number of iterations meets the threshold
            if len(self.possible_locations) > run_settings.RUN_CONFIG.min_customers_for_decomposition and \
                    iterations % run_settings.RUN_CONFIG.decomposition_iterations == 0 and run_start_time:
                logger.info(f"Decomposing solution on iteration {iterations}")
                recombined_solutions = self._decompose_and_solve(algorithm, run_start_time)
                if recombined_solutions:
                    iterations += 1
                    for recombined_solution in recombined_solutions:
                        algorithm.add_to_population(recombined_solution)
                        if algorithm.compare_solution_to_best(best_feasible_solution, recombined_solution):
                            logger.info(f"Reconstructed solution is new best individual:\n{recombined_solution}")
                            best_list.append(best_feasible_solution)
                            if len(best_list) > 2:
                                best_list.pop(0)
                            best_feasible_solution = recombined_solution
                            non_improving_iterations = 0
                    # If none of the recombined solutions are a new best
                    if not non_improving_iterations == 0:
                        non_improving_iterations += 1

        if 0 < self.max_run_time <= (time() - run_start_time):
            stop_reason = "max time limit hit"
        else:
            stop_reason = "max non-improving iteration limit hit"
        logger.info(f"Stopping on iteration {iterations}, because {stop_reason}.")
        logger.info(f"Best solution was found on iteration {best_iteration}")

        # 13: Return the best feasible solution(s)
        if best_feasible_solution:
            if self.decomposition:
                best_list.append(best_feasible_solution)
                return best_list

            return best_feasible_solution
        else:
            # If no feasible solution found, return best infeasible solution.
            penalised_costs = [sol.penalty for sol in algorithm.infeasible_population.population]
            min_penalised_cost = min(penalised_costs)
            return algorithm.infeasible_population.population[penalised_costs.index(min_penalised_cost)]

    # Not implementing multiprocessing now due to time constraints
    # def _run_with_multiprocessing(self, sync_manager: SyncManager, algorithm: HGSADC,
    #                               num_generation_processes: int) -> Optional[Individual]:
    #     """Run the algorithm with multiprocessing and return the best feasible solution.
    #
    #     This method runs the algorithm a bit differently to the sequential algorithm described.
    #
    #     Several processes will be dedicated to the generation, education and repair of individuals. They will share a
    #     lock for adding new individuals to the population."""
    #     logger.info("Initialising multi-process run.")
    #     # The process manager will make sure all processes have access to the same data.
    #     # Any changes to data created via the process manager will be synchronized.
    #
    #     # Assign variables
    #     # best_feasible_solution_dict = sync_manager.dict({})
    #     best_feasible_solution: Optional[Individual] = None
    #     run_information = sync_manager.dict({
    #         "algorithm": algorithm,
    #         "max_non_improving_iterations": self.max_non_improving_iterations,
    #         "max_run_time": self.max_run_time,
    #         # "best_feasible_solution_dict": best_feasible_solution_dict,
    #         "best_feasible_solution": best_feasible_solution,
    #         "iterations": 0,
    #         "non_improving_iterations": 0,
    #         "capacity_feasible_count": 0,
    #         "duration_feasible_count": 0,
    #         "run_status": self.RUN,
    #         "start_time": time()
    #     })
    #     population_lock: multiprocessing.Lock = sync_manager.Lock()
    #
    #     best_random_solution = algorithm.generate_random_individuals(4 * run_settings.RUN_CONFIG.min_pop_size)
    #     if best_random_solution:
    #        # self._update_best_solution_dict(best_random_solution, Individual.parse_dict(best_feasible_solution_dict))
    #         run_information["best_feasible_solution"] = best_random_solution
    #
    #     while run_information["run_status"] != self.STOP and \
    #             run_information["non_improving_iterations"] < run_information["max_non_improving_iterations"] and \
    #             time() - run_information["start_time"] < run_information["max_run_time"]:
    #         if run_information["run_status"] == self.RUN:
    #             with ProcessPoolExecutor() as executor:
    #                 for _ in range(num_generation_processes):
    #                     executor.submit(self._run_single_process, *[run_information, population_lock])
    #
    #         if len(self.possible_locations) > run_settings.RUN_CONFIG.min_customers_for_decomposition and \
    #                 run_information["run_status"] == self.DECOMPOSE:
    #             new_ind = self._decompose_and_solve(algorithm)
    #             run_information["iterations"] += 1
    #             # if algorithm.add_to_population(new_ind,
    #           #                                Individual.parse_dict(run_information["best_feasible_solution_dict"])):
    #             if algorithm.add_to_population(new_ind, run_information["best_feasible_solution"]):
    #                 # run_information["best_feasible_solution_dict"] = new_ind.to_dict()
    #                 run_information["best_feasible_solution"] = new_ind
    #                 run_information["non_improving_iterations"] = 0
    #             else:
    #                 run_information["non_improving_iterations"] += 1
    #             run_information["run_status"] = self.RUN
    #
    #     # return Individual.parse_dict(best_feasible_solution_dict)
    #     return run_information["best_feasible_solution"]
    #
    # @staticmethod
    # def _run_single_process(run_information: dict, population_lock: multiprocessing.Lock):
    #     """This will share similarities with _run_with_single_processing, but is designed to make use of the shared
    #     resources managed by the SyncManager.
    #
    #     Numbered comments indicate correlated lines from Vidal's Algorithm 1."""
    #     # 2: Keep going while the max non-improving iterations and max run time have not been reached
    #     while run_information["run_status"] == Runner.RUN and \
    #             run_information["non_improving_iterations"] < run_information["max_non_improving_iterations"] and \
    #             time() - run_information["start_time"] < run_information["max_run_time"]:
    #         # This is the heavy processing part that each process does individually
    #         # 3, 4, 5 & 6: Create, educate and repair a new individual
    #         child = Runner._create_and_educate_individual(run_information["algorithm"])
    #
    #         # Each process must acquire the lock before it can alter the run information here
    #         # By default the lock will block the process until it is acquired
    #         with population_lock:
    #             # Increment the total number of iterations
    #             run_information["iterations"] += 1
    #             # 6 & 7: Insert the new solution into the appropriate population
    #             # if run_information["algorithm"].add_to_population(child, Individual.parse_dict(
    #             #         run_information["best_feasible_solution_dict"])):
    #             if run_information["algorithm"].add_to_population(child, run_information["best_feasible_solution"]):
    #                 # If the new solution improves on the best feasible,
    #                 # then record it and reset non-improving iterations
    #                 # Runner._update_best_solution_dict(child, run_information["best_feasible_solution_dict"])
    #                 run_information["best_feasible_solution"] = child
    #                 run_information["non_improving_iterations"] = 0
    #             else:
    #                 run_information["non_improving_iterations"] += 1
    #
    #             # 8: Check whether either population has reached the maximum size
    #             if len(run_information["algorithm"].feasible_population.population) >= \
    #                     run_settings.RUN_CONFIG.min_pop_size + run_settings.RUN_CONFIG.generation_size:
    #                 run_information["algorithm"].feasible_population.cull_population(
    #                     run_settings.RUN_CONFIG.min_pop_size)
    #             if len(run_information["algorithm"].infeasible_population.population) >= \
    #                     run_settings.RUN_CONFIG.min_pop_size + run_settings.RUN_CONFIG.generation_size:
    #                 run_information["algorithm"].infeasible_population.cull_population(
    #                     run_settings.RUN_CONFIG.min_pop_size)
    #
    #             # 9: If the best solution hasn't been improved for too long, then diversify the population
    #             if run_information["non_improving_iterations"] % (
    #                     run_settings.RUN_CONFIG.diversification_proportion * run_information[
    #                 "max_non_improving_iterations"]) == 0:
    #                 # run_information["algorithm"].diversify_populations(
    #                 #     Individual.parse_dict(run_information["best_feasible_solution_dict"]))
    #                 run_information["algorithm"].diversify_populations(run_information["best_feasible_solution"])
    #
    #             # Count the number of individuals that are naturally feasible with respect to each type of penalty
    #             if child.capacity_penalty == 0:
    #                 run_information["capacity_feasible_count"] += 1
    #             if child.duration_penalty == 0:
    #                 run_information["duration_feasible_count"] += 1
    #             # 10: Adjust the penalty parameters after a certain number of iterations
    #             if run_information["iterations"] % run_settings.RUN_CONFIG.parameter_adjustment_frequency == 0:
    #                 run_settings.RUN_CONFIG.adjust_penalties(
    #                     run_information[
    #                         "capacity_feasible_count"] / run_settings.RUN_CONFIG.parameter_adjustment_frequency,
    #                     run_information[
    #                         "duration_feasible_count"] / run_settings.RUN_CONFIG.parameter_adjustment_frequency)
    #                 run_information["capacity_feasible_count"] = 0
    #                 run_information["duration_feasible_count"] = 0
    #
    #             # 11: Decompose the problem if the number of iterations meets the threshold
    #             if run_information["iterations"] % run_settings.RUN_CONFIG.decomposition_iterations == 0:
    #                 run_information["run_status"] = Runner.DECOMPOSE

    @staticmethod
    def _update_best_solution_dict(best_feasible_solution: Individual,
                                   best_feasible_solution_dict: Dict[
                                       str, Union[Dict[int, List[Location]], Dict[int, List[int]],
                                                  Tuple[Location, ...], Dict[int, List[Route]], float]]):
        """Updates each key in the best feasible solution dict one by one so that the SyncManager detects the changes.

        I don't want to accidentally overwrite the DictProxy object, so I'm updating each value individually.

        See https://stackoverflow.com/questions/9436757/how-does-multiprocessing-manager-work-in-python"""
        for key, value in best_feasible_solution.attributes_to_dict().items():
            best_feasible_solution_dict[key] = value

    @staticmethod
    def _create_and_educate_individual(algorithm: HGSADC) -> Individual:
        """Create a new solution from two semi-randomly selected solutions from either population."""
        # Create the child
        # logger.debug("Creating child")
        child = algorithm.crossover(*algorithm.select_parents())

        # Possibly educate the child
        # logger.debug("Educating child")
        if random() < run_settings.RUN_CONFIG.education_rate:
            child.educate()

        # Possibly repair the child
        # if not child.feasible and random() < run_settings.RUN_CONFIG.repair_rate:
        #     # logger.debug("Repairing child")
        #     child.educate(10)
        #     if not child.feasible:
        #         child.educate(100)

        # logger.debug(f"Created child: {child}")
        # if child.num_stops > len(data_globals.ALL_CUSTOMERS) * 2:
        #     logger.debug(f"Excessive stops detected. Giant tour: {child.giant_tour_chromosome}")

        return child

    def _decompose_and_solve(self, algorithm: HGSADC, run_start_time: float) -> Optional[List[Individual]]:
        """Decomposes the problem into sub-problems, then creates and runs new Runner objects based on the current
        Runner configuration."""
        problem_components = algorithm.decompose_problem()
        sub_solutions: List[List[Optional[Individual]]] = []
        max_decomp_run_time = self.max_run_time / (len(self.possible_vehicle_types) * 2 + 5)
        # Only run the decomposition if there is enough time left for all the components to reach their max run time
        if max_decomp_run_time * len(problem_components) < self.max_run_time - (time() - run_start_time):
            for index, sub_problem in enumerate(problem_components):
                if sub_problem[0] and len(sub_problem[0]) > 0 and sub_problem[1] and len(sub_problem[1]) > 0:
                    sub_runner = Runner(run_settings.RUN_CONFIG.decomposition_iterations / 2,
                                        max_decomp_run_time,
                                        use_multiprocessing=self.use_multiprocessing, possible_locations=sub_problem[1],
                                        possible_vehicle_types=sub_problem[0], decomposition=True)
                    logger.info(f"Starting decomposed run for sub-problem with {len(sub_problem[1])} customers.")
                    sub_solutions.append(sub_runner.run())
            return algorithm.reconstruct_solutions(sub_solutions)


if __name__ == "__main__":
    """Run the model"""
    run_settings.set_run_data(Data("Model Data.xlsx"))

    start_time = perf_counter()
    runner = Runner(5000, 1800, use_multiprocessing=False)
    best_solution = runner.run()
    end_time = perf_counter()

    if best_solution:
        logger.info("Best solution returned by runner.")
        logger.info(f"{best_solution}")
        logger.info(f"Solution chromosome: {best_solution.giant_tour_chromosome}")
        logger.info(f"Solution routes: {best_solution.routes}")
        logger.info(f"Customers: {runner.possible_locations}")
        logger.info(f"Vehicle types: {runner.possible_vehicle_types}")
        print("Pretty output:\n" + best_solution.pretty_route_output())
        print(f"Run time: {end_time - start_time}")
    else:
        logger.info("No solution returned by runner.")
