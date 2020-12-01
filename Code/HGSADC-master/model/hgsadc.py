"""
Coded by Aaron Shuttleworth in pursuit of a Bachelors of Engineering in Industrial Engineering,
this is an implementation of the Hybrid Genetic Search with Advanced Diversity Control (HGSADC) algorithm put forward by
Vidal et al. in "A Hybrid Genetic Algorithm for Multi-depot and Periodic ArcVehicleType Routing Problems" (2011) and
"A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle_type routing problems with
time-windows" (2012).

First implemented using as reference Vidal et al.'s papers, the model is then adapted to be applicable to the situation
at the SPAR Western Cape DC.

The following Git was used to aid in the implementation of the algorithm
https://github.com/jjuppe/Hybrid-Genetic-Search-with-Adaptive-Diversity-Control
"""
from copy import copy
from random import sample, randint, shuffle
from typing import List, Dict, Union, Tuple, Optional

import numpy

from data_objects import Location, VehicleType
from individual import Individual
from model import run_settings
from population import Population
from util import append_to_chromosome, create_logger, rank_list

logger = create_logger(__name__)


class HGSADC:
    # Assigned on init
    # max_non_improving_iterations: int
    # max_run_time: int
    feasible_population: Population = None
    infeasible_population: Population = None
    possible_locations: numpy.ndarray
    possible_vehicle_types: numpy.ndarray

    def __init__(self, feasible_population: Population, infeasible_population: Population,
                 possible_locations: numpy.ndarray, possible_vehicle_types: numpy.ndarray,
                 allow_split_deliveries: bool = True):
        """Initialize the instance of the algorithm"""

        # self.max_non_improving_iterations = max_non_improving_iterations
        # self.max_run_time = max_run_time
        self.feasible_population = feasible_population
        self.infeasible_population = infeasible_population
        self.possible_locations = possible_locations
        self.possible_vehicle_types = possible_vehicle_types
        self.allow_split_deliveries = allow_split_deliveries
        # self.return_solutions = return_solutions

    def generate_random_individuals(self, new_individuals: int,
                                    best_feasible_solution: Union[None, Individual] = None) -> Union[None, Individual]:
        """Generate a specified number of random solutions. Returns the best feasible solution, if any are found."""
        logger.info(f"Generating {new_individuals} random solutions")

        for _ in range(new_individuals):
            new_ind = Individual.create_random_solution(self.possible_locations, self.possible_vehicle_types)
            # logger.debug(f"Generated individual {new_ind}")
            new_ind.evaluate()
            # if new_ind.feasible:
            #     self.feasible_population.append(new_ind)
            # else:
            #     self.infeasible_population.append(new_ind)
            self.add_to_population(new_ind)
            if self.compare_solution_to_best(best_feasible_solution, new_ind):
                best_feasible_solution = new_ind

        return best_feasible_solution

    @staticmethod
    def compare_solution_to_best(best_feasible_solution: Optional[Individual], new_solution: Individual):
        return new_solution.feasible and (
                best_feasible_solution is None or new_solution.cost < best_feasible_solution.cost)

    def select_parents(self) -> Tuple[Individual, Individual]:
        """Select two parents from the population to crossover"""
        # Parents are selected from both populations combined
        combined_population = Population(
            population=[*self.feasible_population.population, *self.infeasible_population.population])
        biased_fitnesses = combined_population.get_biased_fitness_list()

        # Select four random members of the population
        parent_indices = sample(range(len(combined_population.population)), 4)
        # parent_1, parent_2 = None, None
        # The members are paired, and the individual with the lowest fitness (highest rank) of each pair is selected
        if biased_fitnesses[parent_indices[0]] < biased_fitnesses[parent_indices[1]]:
            parent_1 = combined_population.population[parent_indices[0]]
        else:
            parent_1 = combined_population.population[parent_indices[1]]
        if biased_fitnesses[parent_indices[2]] < biased_fitnesses[parent_indices[3]]:
            parent_2 = combined_population.population[parent_indices[2]]
        else:
            parent_2 = combined_population.population[parent_indices[3]]

        return parent_1, parent_2

    def crossover(self, parent_1: Individual, parent_2: Individual) -> Individual:
        """Crossover two parents using the Periodic crossover with Intersections (PIX) algorithm,
        as defined in Vidal et al. (2011)."""
        # logger.debug("Beginning crossover")
        # logger.debug(f"Par 1 chrom: {parent_1.giant_tour_chromosome}")
        # logger.debug(f"Par 2 chrom: {parent_2.giant_tour_chromosome}")
        child_vehicle_type_chromosome: Union[None, Dict[int, List[int]]] = {}
        child_giant_tour_chromosome: Dict[int, List[Location]] = {}
        for vehicle_type in self.possible_vehicle_types:
            child_giant_tour_chromosome[vehicle_type.data_index] = []

        num_vehicle_types = len(self.possible_vehicle_types)

        # Step 0: Inheritance Rule
        # Figure out which tours are inherited from P1, which from P2 and which others are mixed
        # In Vidal's paper, the tours are per-period and per-depot, so each pairing needs to be accounted for.
        # In my implementation, there are only the vehicle_index types to account for.
        if num_vehicle_types > 1:
            cuts = sample(range(num_vehicle_types), 2)
            if cuts[0] < cuts[1]:
                n_1 = cuts[0]
                n_2 = cuts[1]
            else:
                n_1 = cuts[1]
                n_2 = cuts[0]

            # Put the vehicle_index types into a randomly ordered list
            vehicle_type_list = list(self.possible_vehicle_types)
            shuffle(vehicle_type_list)

            # Divide up the vehicle_index types into the sets to be inherited
            set_1 = vehicle_type_list[0:n_1]
            set_2 = vehicle_type_list[n_1:n_2]
            set_mix = vehicle_type_list[n_2:num_vehicle_types]
        else:
            # If there is only one vehicle type, it goes into the mixed set
            set_1 = []
            set_2 = []
            set_mix = list(self.possible_vehicle_types)
        # logger.debug(f"Crossover sets: {set_1},\t{set_2},\t{set_mix}")

        # Step 1: Inherit data from P1
        # Set 1 inherits the whole tours
        for vehicle_type in set_1:
            if parent_1.giant_tour_chromosome.get(vehicle_type.data_index) is None:
                continue

            child_giant_tour_chromosome[vehicle_type.data_index] = \
                [copy(stop) for stop in parent_1.giant_tour_chromosome[vehicle_type.data_index]]
            for child in child_giant_tour_chromosome[vehicle_type.data_index]:
                append_to_chromosome(child_vehicle_type_chromosome, child.data_index, vehicle_type, False)
        # Mixed set inherits pieces of the tours
        for vehicle_type in set_mix:
            if parent_1.giant_tour_chromosome.get(vehicle_type.data_index) is None:
                continue

            tour_length = len(parent_1.giant_tour_chromosome[vehicle_type.data_index])

            cut_a = randint(0, tour_length)
            cut_b = cut_a
            if tour_length > 1:
                while cut_b == cut_a:
                    # logger.debug(f"{tour_length}: {cut_a}, {cut_b}")
                    cut_b = randint(0, tour_length)

            if cut_a < cut_b:
                child_tour = [copy(stop) for stop in
                              parent_1.giant_tour_chromosome[vehicle_type.data_index][cut_a:cut_b]]
            else:
                child_tour = [copy(stop) for stop in [*parent_1.giant_tour_chromosome[vehicle_type.data_index][0:cut_b],
                                                      *parent_1.giant_tour_chromosome[vehicle_type.data_index][
                                                       cut_a:tour_length]]]

            child_giant_tour_chromosome[vehicle_type.data_index] = child_tour

            for child in child_giant_tour_chromosome[vehicle_type.data_index]:
                append_to_chromosome(child_vehicle_type_chromosome, child.data_index, vehicle_type, False)

        # logger.debug(f"Data inherited from P1")
        # all_stops = [stop_t for vehicle_type_t in [*set_1, *set_mix] for stop_t in
        #              parent_1.giant_tour_chromosome.get(vehicle_type_t.data_index)]
        # if len(child_giant_tour_chromosome.keys()) == 0 and len(all_stops) > 0:
        #     raise ValueError(f"No keys in child chromosome. All stops {all_stops}")

        # Step 2: Inherit data from P2
        # For all vehicle_index types in set 2 and mix, loop through the customers in parent 2.
        # If the customer either has a matching vehicle_index type or none in the child's vehicle_index chromosome,
        # add it to this type's tour. Recall that the vehicle_index types are in a random order in the sets.
        # This allows customers to be reassigned between vehicle_index types, while vaguely keeping the parent's order.
        for vehicle_type in [*set_2, *set_mix]:
            if parent_2.giant_tour_chromosome.get(vehicle_type.data_index):
                for stop in parent_2.giant_tour_chromosome[vehicle_type.data_index]:
                    customer_vehicles = child_vehicle_type_chromosome.get(stop.data_index)
                    if customer_vehicles is None or vehicle_type in customer_vehicles:
                        append_to_chromosome(child_giant_tour_chromosome, vehicle_type.data_index, stop)
                        append_to_chromosome(child_vehicle_type_chromosome, stop.data_index, vehicle_type, False)

        # logger.debug(f"Data inherited from P2")

        # logger.debug(f"Child chrom: {child_giant_tour_chromosome}")

        # Step 3: Complete customer services
        child = Individual(child_giant_tour_chromosome, vehicle_type_chromosome=child_vehicle_type_chromosome,
                           possible_locations=self.possible_locations)
        # child.update_all_routes_location_indices()
        # child.complete_customer_services(self.allow_split_deliveries)

        # if len(child_giant_tour_chromosome.keys()) == 0:
        #     raise ValueError(f"No keys in child chromosome.")

        # child.complete_customer_services_slow()
        child.evaluate()

        # logger.debug(f"Completed customer services")

        # Create and return the child
        return child

    # def add_to_population(self, child: Individual, best_feasible_solution: Union[None, Individual],
    #                       allow_repair: bool = True) -> Optional[Individual]:
    def add_to_population(self, child: Individual):
        """Adds the provided individual to the appropriate population."""
        if child.feasible:
            # If the new solution is feasible, add it to the feasible population and compare it to the best feasible
            self.feasible_population.append(child)
        else:
            # Otherwise, insert it into infeasible population
            self.infeasible_population.append(child)

    def diversify_populations(self, best_feasible_solution: Union[None, Individual]):
        """Diversify population by culling both to u/3 individuals, then generating u*4 new individuals."""
        survivor_count = int(run_settings.RUN_CONFIG.min_pop_size / 3)
        self.feasible_population.cull_population(survivor_count)
        self.infeasible_population.cull_population(survivor_count)
        self.generate_random_individuals(run_settings.RUN_CONFIG.min_pop_size * 4,
                                         best_feasible_solution=best_feasible_solution)

    def decompose_problem(self) -> List[Tuple[Tuple[VehicleType, ...], Tuple[Location, ...]]]:
        """Decompose the problem into subproblems.

        This is done by selecting one elite solution from the best 25% feasible solutions (or infeasible if there are no
        feasible), then breaking its giant tour chromosome up by vehicle_index type.
        These are returned, and a new HGSADC instance is run to solve each."""
        # Use feasible population if there are solutions within it
        if len(self.feasible_population.population) > 0:
            target_pop = self.feasible_population
        else:
            target_pop = self.infeasible_population
        # Rank the solutions
        biased_fitnesses = target_pop.get_biased_fitness_list()
        ranks = rank_list(biased_fitnesses)
        # Chose a random rank in the top 25%
        selected_rank = randint(0, int(len(target_pop.population) / 4))
        # Find the individual with this rank
        index = ranks.index(selected_rank)
        decompose_individual: Individual = target_pop.population[index]
        # This will be used as the giant tour chromosome for the reconstructed individual
        # new_giant_tour: Dict[int, List[ArcLocation]] = {}
        decomposed_components = []
        customer_count = 0
        sub_types: List[VehicleType] = []
        sub_locations: Dict[int, Location] = {}

        # For each vehicle_index type, solve a subproblem with only the stores in that type
        for vehicle_type, tour in decompose_individual.giant_tour_chromosome.items():
            sub_types.append(VehicleType(vehicle_type))
            self._generate_decomposition_location_dict(tour, sub_locations)
            customer_count += len(tour)

            if customer_count > run_settings.RUN_CONFIG.min_customers_for_decomposition:
                decomposed_components.append((tuple(sub_types), tuple(sub_locations.values())))
                customer_count = 0
                sub_types = []
                sub_locations = {}

        return decomposed_components

    @staticmethod
    def _generate_decomposition_location_dict(tour: List[Location], locations: Dict[int, Location]):
        """Loops through a tour and creates a dict of possible locations. Each location in the tuple will have its
        demand set to the total of the serviced demand for all stops in the tour."""
        for location in tour:
            index = location.data_index
            if locations.get(index):
                locations[index].demand += location.serviced_demand
            else:
                locations[index] = Location(location.data_index, demand=location.serviced_demand)

        return locations

    @staticmethod
    def reconstruct_solution(sub_solutions: List[Individual]) -> Individual:
        """Reconstructs a solution from a number of solutions to decomposed problems."""
        reconstructed_giant_tour_chromosome = {}

        for solution in sub_solutions:
            if solution:
                for vehicle_type, tour in solution.giant_tour_chromosome.items():
                    reconstructed_giant_tour_chromosome[vehicle_type] = tour

        return Individual(reconstructed_giant_tour_chromosome)

    @staticmethod
    def reconstruct_solutions(sub_solutions: List[List[Optional[Individual]]]) -> List[Individual]:
        return [HGSADC.reconstruct_solution(sub_solution) for sub_solution in sub_solutions]
