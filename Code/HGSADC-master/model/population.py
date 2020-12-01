from math import inf
from multiprocessing.managers import SyncManager
from typing import List, Dict

from individual import Individual
from model import run_settings
from util import rank_list, create_logger

logger = create_logger(__name__)


class Population:
    """Sub-class of list intended to hold a population of individuals, a Hamming distance matrix and some helper
    functions.

    The hamming distance matrix will be automatically updated when individuals are appended or popped from the array."""
    process_manager: SyncManager
    population: List[Individual]
    hamming_matrix: List[List[float]]

    def __init__(self, population: List[Individual] = None, process_manager: SyncManager = None):
        """Creates a matrix with the Hamming distances between all members of the population.

        If a process manager is provided, the object uses it to allow changes to the population to be synchronised
        across processes."""
        self.process_manager = process_manager

        if process_manager:
            self.population = process_manager.list()
            self.hamming_matrix = process_manager.list()
        else:
            self.population = []
            self.hamming_matrix = []

        if population:
            self.population = population

        # Find the distance between each member of the population
        for i, ind_1 in enumerate(self.population):
            if self.process_manager:
                self.hamming_matrix.append(self.process_manager.list())
            else:
                self.hamming_matrix.append([])
            for j, ind_2 in enumerate(self.population):
                if i < j:
                    # Since j is incremented before i, the first time each pairing is found will be when i < j.
                    self.hamming_matrix[i].append(
                        self.normalized_distance(ind_1.vehicle_type_chromosome, ind_2.vehicle_type_chromosome))
                elif i > j:
                    # If i > j, then this pairing has already been calculated in the opposite direction
                    self.hamming_matrix[i].append(self.hamming_matrix[j][i])
                else:
                    # An individual's distance to itself will always be 0
                    self.hamming_matrix[i].append(0)

    def pop(self, __index: int = ...) -> Individual:
        """Removes a specific individual from the population and matrix."""
        # if __index >= len(self.population):
        #     raise IndexError(f"Index {__index} not within population of size {len(self.population)}")

        # Remove the row for this index
        self.hamming_matrix.pop(__index)

        # Remove the column for this index
        for i, row in enumerate(self.hamming_matrix):
            row.pop(__index)

        # Remove the individual from the population
        return self.population.pop(__index)

    def append(self, __object: Individual) -> None:
        """Adds a new individual to the population and matrix."""
        # Create a new row for this individual's distance to others
        if self.process_manager:
            new_distances = self.process_manager.list(
                [self.normalized_distance(__object.vehicle_type_chromosome, individual_2.vehicle_type_chromosome) for
                 individual_2 in self.population])
        else:
            new_distances = [
                self.normalized_distance(__object.vehicle_type_chromosome, individual_2.vehicle_type_chromosome) for
                individual_2 in self.population]
        new_distances.append(0)

        # Add an extra column for others' distances to this
        for i, row in enumerate(self.hamming_matrix):
            row.append(new_distances[i])

        # Add the new row to the matrix
        self.hamming_matrix.append(new_distances)

        # Add the individual to the population
        self.population.append(__object)

    def remove(self, __object: Individual):
        """Tries to remove the given Individual from the population and matrix."""
        # Find the given individual's index
        index = self.population.index(__object)
        # Remove the object at that index
        self.pop(index)

    @staticmethod
    def normalized_distance(chromosome_1: Dict[int, List[int]], chromosome_2: Dict[int, List[int]]) -> float:
        """Find the Hamming distance between two individuals then divide it by the number of stores.

        The Hamming distance is defined as the number of items you need to change in one string/list to make it
        identical to the other.

        Vidal's paper indicates to use the pattern (period) and depot chromosomes for this. Any stores that don't have
        matching periods or depots are counted.
        My implementation will thus use the vehicle_index type assignment of the stores for the distance.

        I am curious about the merits (or lack thereof) of concatenating all patterns in the giant tour chromosome and
        running a Hamming matrix on this."""
        if len(chromosome_1) == 0:
            return inf

        return sum(types_1 != types_2 for types_1, types_2 in zip(chromosome_1, chromosome_2)) / len(chromosome_1)

        # return sum(value != chromosome_2[key] for key, value in chromosome_1.items()) / len(chromosome_1)

    def get_diversity_contribution_list(self) -> List[float]:
        """Returns a list of the diversity contribution of all individuals."""
        diversity_list = []

        for i in range(len(self.population)):
            # Get the row for this individual's relation to all others
            distances = self.hamming_matrix[i].copy()
            # Sort the values in ascending order
            distances.sort()
            # Calculate the diversity population based on n_close closest individuals, aside from itself
            diversity_list.append(
                sum(distances[1:run_settings.RUN_CONFIG.n_close + 1]) / run_settings.RUN_CONFIG.n_close)

        return diversity_list

    def get_biased_fitness_list(self) -> List[float]:
        """Returns a list of the biased fitness of all individuals. 
        This accounts for each individual's cost rank and diversity rank."""
        # If the population size is closer to the number of elite individuals kept between generations,
        # then diversity is weighed less importantly.
        if len(self.population) > 0:
            bias = (1 - (run_settings.RUN_CONFIG.min_pop_size / len(self.population)))
            cost_list = [ind.get_penalised_cost() for ind in self.population]
            fitness_ranks = rank_list(cost_list)
            diversity_ranks = rank_list(self.get_diversity_contribution_list(), ascending=False)

            return [fitness_ranks[i] + bias * diversity_ranks[i] for i in range(len(self.population))]
        else:
            return []

    def cull_population(self, num_survivors: int):
        """Cull a population down to the best few individuals."""
        # Find which individuals are considered clones of others
        clones = self.find_clones()
        # Rank the population according to biased fitness
        biased_fitness_ranks = rank_list(self.get_biased_fitness_list())
        # Sort the population based on the ranks
        sorted_population = [individual for _, individual in sorted(zip(biased_fitness_ranks, self.population))]
        current_index = len(sorted_population) - 1

        # Remove individuals from population until target number of individuals is met
        while len(self.population) > num_survivors:
            current_individual = sorted_population[current_index]
            # If there are still clones
            if len(clones) > 0:
                # Check if the current individual is a clone
                if clones.count(current_individual) > 0:
                    # If so, remove it
                    self.remove(current_individual)
                    clones.remove(current_individual)
                    sorted_population.remove(current_individual)
                    # If it was the last clone, reset the current index
                    if len(clones) == 0:
                        current_index = len(sorted_population)
            else:
                # If not, just remove the current individual
                self.remove(current_individual)
                sorted_population.remove(current_individual)

            current_index -= 1

    def find_clones(self) -> List[Individual]:
        """Finds any individuals that have the same cost or a hamming distance of 0."""
        clones = []
        # Loop through to find all the clones and compile a list of them
        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i < j:
                    ind_i: Individual = self.population[i]
                    ind_j: Individual = self.population[i]
                    if ind_i.get_penalised_cost() == ind_j.get_penalised_cost() or self.hamming_matrix[i][j] == 0:
                        # Append i because it's guaranteed to be in ascending order, whereas j is not
                        clones.append(ind_j)
                        if clones.count(ind_i) == 0:
                            clones.append(ind_i)
                else:
                    break

        return clones
