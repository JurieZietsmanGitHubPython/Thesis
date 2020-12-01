"""Store information about individual solutions to the VRP."""
import math
from collections import Counter
from copy import deepcopy, copy
from random import shuffle, sample, random
from time import time
from typing import List, Dict, Union, Tuple, Optional

import numpy

from data_objects import Location, LocationChecker, VehicleType, data_globals
from model import run_settings
from util import append_to_chromosome, create_logger

logger = create_logger(__name__)


class Route:
    """Stores a route belonging to a solution"""

    def __init__(self, sequence: List[Location], vehicle_type: VehicleType):
        """Initialise the route object"""
        self.load: Optional[int] = None
        self.distance_travelled: Optional[float] = None
        self.driving_time: Optional[float] = None
        self.rest_time: Optional[float] = None
        self.wait_time: Optional[float] = None
        self.service_time: Optional[float] = None
        self.warp_time: Optional[float] = None
        self.departure_time: Optional[float] = None
        # self.alt_departure_time: Optional[float] = None
        self.windowed: Optional[bool] = None
        self.total_time: Optional[float] = None
        self.cost: Optional[float] = None
        self.duration_penalty: Optional[float] = None
        self.capacity_penalty: Optional[float] = None
        self.third_party_only: bool = False

        self.sequence = sequence
        self.vehicle_type = vehicle_type
        self.evaluate()

    def __repr__(self):
        return f"Route with sequence {self.sequence}"

    # def __copy__(self):
    #     cls = self.__class__
    #     new_object = cls.__new__(cls)
    #     new_object.__dict__.update(self.__dict__)

    def __deepcopy__(self, memo: dict):
        """Deep copy is the same as shallow copy, but also shallow copies the ArcLocation objects in its sequence."""
        # Check whether a copy has already been made
        existing = memo.get(self)
        if existing:
            return existing

        # Perform shallow copy of self
        # cls = self.__class__
        # new_object = cls.__new__(cls)
        # new_object.__dict__.update(self.__dict__)
        new_object = copy(self)
        new_object.sequence = [copy(stop) for stop in self.sequence]

        # Add the new copy to the memo and return it
        memo[self] = new_object
        return new_object

    def evaluate(self, extra_penalty: int = 1) -> float:
        """Determine all the costs and penalties for this route.

        Departure time is decided based on a greedy heuristic that makes the vehicle_index arrive at the window start of
         the first store with a window."""

        capacity = self.vehicle_type.capacity
        distance_cost = self.vehicle_type.distance_cost
        time_cost = self.vehicle_type.time_cost

        # Important properties of this route to determine the cost
        # Total load for all customers
        self.load: int = 0
        # Distance (Km) travelled between all locations (incl depot)
        self.distance_travelled: float = 0
        # Driving time (H) spent travelling between all locations (incl depot)
        self.driving_time: float = 0
        # Time (H) spent resting
        self.rest_time: float = 0
        # Time (H) spent waiting for service windows to begin (depot N/A)
        self.wait_time: float = 0
        # Time (H) spent unloading at locations (excl depot)
        self.service_time: float = 0
        # Total time (H) which the vehicle_index missed service windows by
        self.warp_time: float = 0
        # Time (H) at which the vehicle_index departs in order to service the route
        self.departure_time: float = 0
        # self.alt_departure_time: float = 0
        # The first store that is found with a service window will be used to set the service time.
        self.windowed: bool = False
        # Time (H) at which the vehicle_index visits each location
        # self.visit_times: List[float] = []
        if len(self.sequence) == 0:
            self.total_time = 0
            self.cost = 0
            self.duration_penalty = 0
            self.capacity_penalty = 0
            return 0

        # Grab some values from the configuration so that they do not have to be repeatedly retrieved
        depot = data_globals.DEPOT
        window_threshold = run_settings.RUN_CONFIG.window_threshold
        consider_rest_time = run_settings.RUN_CONFIG.consider_rest_time
        continuous_drive_limit = run_settings.RUN_CONFIG.continuous_drive_limit
        continuous_rest = run_settings.RUN_CONFIG.continuous_rest
        daily_drive_limit = run_settings.RUN_CONFIG.daily_drive_limit
        daily_rest = run_settings.RUN_CONFIG.daily_rest
        local_range = run_settings.RUN_CONFIG.local_range
        return_to_depot_before = run_settings.RUN_CONFIG.return_to_depot_before
        duration_penalty_multiplier = run_settings.RUN_CONFIG.duration_penalty_multiplier
        capacity_penalty_multiplier = run_settings.RUN_CONFIG.capacity_penalty_multiplier

        # The first "previous customer" is the depot itself
        previous_customer = depot

        # Loop through the sequence and calculate the distance and times for the travel
        for index, customer in enumerate(self.sequence):
            self.load += customer.serviced_demand

            arc_time, wait, serve, warp = 0.0, 0.0, 0.0, 0.0
            # Find the travel distance and time from the previous customer (or the depot)
            self.distance_travelled += previous_customer.travel_distances[customer.data_index]
            # Driving is added to the sums before checking the windows, as the vehicle_index first needs to arrive
            # Arc time needs to be stored separately for the continuous driving time check
            arc_time = previous_customer.travel_times[customer.data_index]
            self.driving_time += arc_time

            # If any customer is further than the local range from the DC, then the route must be served by third-party
            # vehicles
            if not self.third_party_only and depot.travel_distances[customer.data_index] > local_range:
                self.third_party_only = True

            # serve += run_settings.RUN_DATA.average_unload_time[customer] * self.sequence[index].serviced_demand
            serve += customer.expected_unload_time

            # Allow 45 min rest for each 5 continuous hours of driving
            if arc_time >= continuous_drive_limit and consider_rest_time:
                self.rest_time += continuous_rest * math.floor(arc_time / continuous_drive_limit)

            # Check if a window has been found
            if self.windowed:
                # If one has been found, calculate any wait or warp time for this store
                wait = max(0.0, customer.window_start -
                           (self.departure_time + self.driving_time + self.rest_time + self.wait_time +
                            self.service_time - self.warp_time))
                warp = max(0.0, (self.departure_time + self.driving_time + self.rest_time + self.wait_time +
                                 self.service_time - self.warp_time) - customer.window_end)
                if wait > 0 and warp > 0:
                    raise ValueError(f"Wait {wait} and warp {warp} both greater than 0!")
            elif customer.window_end - customer.window_start <= window_threshold:
                # If one hasn't been found, check if this store has a window. If the store does have a window,
                # set departure time so that the vehicle_index will arrive at window end
                # There won't be any wait or warp time yet, as those are only summed after the first windowed store
                self.windowed = True
                self.departure_time = customer.window_end - (self.driving_time + self.rest_time + self.service_time)
                # self.alt_departure_time = customer.window_start - self.driving_time + self.rest_time + self.service_time

            # Once it has been determined if the vehicle_index will need to wait for window start, warp to window end,
            # then add these to the sums, along with the service time at this store
            self.wait_time += wait
            self.service_time += serve
            self.warp_time += warp

            # After visiting the last store, the vehicle_index must also return to the depot.
            if index == len(self.sequence) - 1:
                self.distance_travelled += customer.travel_distances[depot.data_index]
                self.driving_time += customer.travel_times[depot.data_index]
            else:
                previous_customer = customer

        # If the total driving time exceeds the daily limit, then rest time for an 8 hr sleep
        if self.driving_time >= daily_drive_limit and consider_rest_time:
            self.rest_time += daily_rest * math.floor(self.driving_time / daily_drive_limit)

        # The total time is the driving time plus the service and time
        # Warp time is subtracted from total time
        self.total_time = self.driving_time + self.service_time + self.wait_time - self.warp_time + self.rest_time

        # If the vehicle must return to the depot before a certain time, warp it if it is not back before that time
        if return_to_depot_before > 0:
            warp_change = max(0.0, self.departure_time + self.total_time - return_to_depot_before)
            if warp_change > 0:
                self.warp_time += warp_change
                self.total_time -= warp_change

        # Cost without extra penalties
        self.cost = (self.total_time * time_cost) + (self.distance_travelled * distance_cost)

        # Penalties are applied if total time is greater than limit, time is warped, or load is greater than capacity
        self.duration_penalty = duration_penalty_multiplier * self.warp_time
        self.capacity_penalty = capacity_penalty_multiplier * max(0, self.load - capacity)

        # self.feasible = self.penalty == 0
        return self.get_penalised_cost(extra_penalty)

    @property
    def penalty(self) -> float:
        return self.duration_penalty + self.capacity_penalty

    def get_penalised_cost(self, extra_penalty: int) -> float:
        return self.cost + (extra_penalty * self.penalty)


class Individual:
    """Stores an individual solution to the problem"""

    def __init__(self, giant_tour_chromosome: Dict[int, List[Location]],
                 vehicle_type_chromosome: Optional[Dict[int, List[int]]] = None,
                 possible_locations: numpy.ndarray = data_globals.ALL_CUSTOMERS):
        """Create the individual with specified chromosomes.
        If no vehicle_index type chromosome is provided, one will be generated."""
        self.routes: Dict[int, List[Route]] = {}
        # self.hired_vehicles: Dict[int, int] = {}
        self.cost: Optional[float] = None
        self.capacity_penalty: Optional[float] = None
        self.duration_penalty: Optional[float] = None

        # logger.debug(f"Creating individual with giant tour {giant_tour_chromosome}")
        # This is used for the complete_customer_services method
        self.possible_locations = possible_locations

        # Giant tour indicates all the customers that are visited from a particular depot in a particular period
        # Any customers visited during the same trip remain the same,
        # but trips are concatenated and visits to the depots are removed from the giant tour
        # The giant tour must consist of a dict layered upon a dict for each other chromosome
        # Repetitions of stores in a tour mean the load is split
        self.giant_tour_chromosome = giant_tour_chromosome
        # The giant tour chromosome will be split into sets of individual routes

        # if len(vehicle_type_chromosome) != len(demand_chromosome):
        #     raise ValueError("All minor chromosomes must be the same length.")
        # logger.debug(f"Giant tour chromosome assigned: {self.giant_tour_chromosome} ")

        # Minor Chromosomes
        # Which vehicle_index types are delivering to any given store.
        if vehicle_type_chromosome:
            self.vehicle_type_chromosome: Dict[int, List[int]] = vehicle_type_chromosome
        else:
            self._construct_vehicle_type_chromosome()

        # This tracks how the demand of stores is split between types and same-type repetions. The amount delivered
        # to a store across all periods must still match the total demand. Dict represents the vehicle_index type,
        # while the list stores the matching demand for each customer in the tour.
        # self.demand_chromosome = demand_chromosome
        # NOTE: Demand chromosome replaced with giant tour storing ArcLocation objects

        # Create the routes dictionary
        for vehicle_type in giant_tour_chromosome.keys():
            self.routes[vehicle_type] = []

    def __repr__(self) -> str:
        if self.feasible:
            ret_str = "Feasible"
        else:
            ret_str = "Infeasible"
        ret_str += f" Individual with {self.num_stops} stops and cost {self.cost}"
        if not self.feasible:
            ret_str += f" and penalty {self.penalty}"

        return ret_str

    def attributes_to_dict(self) -> Dict[
        str, Union[Dict[int, List[Location]], Dict[int, List[int]], numpy.ndarray,
                   Dict[int, List[Route]], float]]:
        """Convert all attributes of self into a dictionary."""
        return {
            "giant_tour_chromosome": self.giant_tour_chromosome,
            "vehicle_type_chromosome": self.vehicle_type_chromosome,
            "possible_locations": self.possible_locations,
            "routes": self.routes,
            "cost": self.cost,
            "capacity_penalty": self.capacity_penalty,
            "duration_penalty": self.duration_penalty
        }

    @staticmethod
    def parse_dict(individual_dict: Dict[str, Union[Dict[int, List[Location]], Dict[int, List[int]], numpy.ndarray,
                                                    Dict[int, List[Route]], float]]) -> "Individual":
        """Convert a dict containing the attributes of an individual into a new individual,"""
        # Create the individual with the required attributes
        if individual_dict.get("giant_tour_chromosome") and individual_dict.get("vehicle_type_chromosome") and \
                individual_dict.get("possible_locations"):
            parsed_individual = Individual(giant_tour_chromosome=individual_dict["giant_tour_chromosome"],
                                           vehicle_type_chromosome=individual_dict["vehicle_type_chromosome"],
                                           possible_locations=individual_dict["possible_locations"])
        else:
            raise ValueError("The provided dictionary doesn't have the required keys.")

        # Assign the other attributes if they are present in the dict
        if individual_dict.get("routes"):
            parsed_individual.routes = individual_dict["routes"]
        if individual_dict.get("cost"):
            parsed_individual.cost = individual_dict["cost"]
        if individual_dict.get("capacity_penalty"):
            parsed_individual.capacity_penalty = individual_dict["capacity_penalty"]
        if individual_dict.get("duration_penalty"):
            parsed_individual.duration_penalty = individual_dict["duration_penalty"]

        return parsed_individual

    def pretty_route_output(self) -> str:
        """Outputs the routes into a string that is more easily readable by humans."""
        pretty_output = ""
        for vehicle_type_index, tour in self.routes.items():
            vehicle_type = VehicleType(vehicle_type_index)
            pretty_output += f"{vehicle_type.name} (capacity {vehicle_type.capacity}):\n"
            for route in tour:
                stops = [f"{customer.name} ({customer.serviced_demand})" for customer in route.sequence]

                pretty_output += " -> ".join(stops) + "\n"
        return pretty_output

    def routes_to_dict(self) -> Optional[Dict[int, List[List[List[int]]]]]:
        """Outputs the routes into a JSON string"""
        route_dict = {}
        try:
            for vehicle_type, tour in self.routes.items():
                route_dict[vehicle_type] = [
                    [[int(stop.data_index), int(stop.serviced_demand)] for stop in route.sequence] for route in tour]
            return route_dict
        except TypeError:
            return None

    @staticmethod
    def create_random_solution(possible_locations: numpy.ndarray,
                               included_vehicle_types: numpy.ndarray) -> "Individual":
        """Creates a random new individual.
        A random giant tour chromosome is generated then used to create the individual."""
        # Generate the chromosome
        giant_tour: Dict[int, List[Location]] = {}
        # Make a deep copy so that the individuals don't share location objects
        # This is important, since each location object also stores the serviced amount
        locations = list(possible_locations)
        shuffle(locations)

        # The locations need to be randomly divided between the different vehicle_index types.
        # Get the available vehicle_index types, and give each of them a proportion according to
        # (num vehicles * capacity * random multiplier) / total
        # Random multiplier will be between 0 and 2
        type_proportions = [vehicle_type.available_vehicles * vehicle_type.capacity * (random() * 2) for vehicle_type in
                            included_vehicle_types]
        total_randomised_capacity = sum(type_proportions)
        for i in range(len(type_proportions)):
            type_proportions[i] = type_proportions[i] / total_randomised_capacity

        # Now divide the randomly-ordered locations amongst the vehicle_index types
        last_index = 0
        # logger.debug(f"{included_vehicle_types}:\t{type_proportions}")
        for type_index, vehicle_type in enumerate(included_vehicle_types):
            next_index = last_index + math.ceil(type_proportions[type_index] * len(locations))
            # If nearly at end of the vehicle_index types, make sure that the final index includes the last customer
            if next_index > len(locations):
                next_index = len(locations)
            # Take a slice of the customer list. The locations are already in a random order.
            giant_tour[vehicle_type.data_index] = locations[last_index:next_index]
            last_index = next_index

        return Individual(giant_tour_chromosome=giant_tour, possible_locations=possible_locations)

    @staticmethod
    def reconstruct_solution(solution_dict: Dict[int, List[List[Union[Tuple[int, int], Location, List[int]]]]],
                             allow_completion: bool = True) -> "Individual":
        """Reconstructs an individual from a dict containing representations of stops within routes within tours."""
        giant_tour_chromosome: Dict[int, List[Location]] = {}
        solution_routes: Dict[int, List[Route]] = {}

        serviced_demand: List[int] = [0 for _ in data_globals.ALL_CUSTOMERS]

        for vehicle_index, int_tour in solution_dict.items():
            giant_tour_chromosome[vehicle_index] = []
            solution_routes[vehicle_index] = []

            for int_route in int_tour:
                loc_route = []
                for stop in int_route:
                    if isinstance(stop, (tuple, list)):
                        stop_loc = Location(*stop)
                    elif isinstance(stop, Location):
                        stop_loc = stop
                    else:
                        raise TypeError(f"Inappropriate stop data type! {type(stop)}, {stop}")

                    # If this stop is at the same location as the previous stop, merge them into one stop
                    if len(loc_route) > 0 and loc_route[-1].data_index == stop_loc.data_index:
                        loc_route[-1].serviced_demand += stop_loc.serviced_demand
                        customer_index = numpy.where(data_globals.ALL_CUSTOMERS == loc_route[-1])[0][0]
                        serviced_demand[customer_index] += stop_loc.serviced_demand
                        continue

                    if stop_loc.serviced_demand == 0:
                        stop_loc.serviced_demand = 1

                    loc_route.append(stop_loc)
                    giant_tour_chromosome[vehicle_index].append(stop_loc)

                    # Add the serviced demand for this stop to the total for the customer
                    # found_locations = numpy.where(data_globals.ALL_CUSTOMERS == loc_route[-1])[0]
                    # if len(found_locations) > 0:
                    #     customer_index = found_locations[0]
                    #     serviced_demand[customer_index] += loc_route[-1].
                    customer_index = numpy.where(data_globals.ALL_CUSTOMERS == loc_route[-1])[0][0]
                    serviced_demand[customer_index] += loc_route[-1].serviced_demand

                if len(loc_route) > 0:
                    route = Route(loc_route, VehicleType(vehicle_index))
                    solution_routes[vehicle_index].append(route)

        reconstructed_individual = Individual(giant_tour_chromosome)
        reconstructed_individual.routes = solution_routes
        reconstructed_individual.evaluate(allow_completion)

        return reconstructed_individual

    def _construct_vehicle_type_chromosome(self):
        """Constructs the vehicle_index type chromosome from the giant tour chromosome"""
        self.vehicle_type_chromosome: Dict[int, List[int]] = {}
        for vehicle_type, tour in self.giant_tour_chromosome.items():
            for customer in tour:
                append_to_chromosome(self.vehicle_type_chromosome, customer.data_index, vehicle_type, False)
                # if self.vehicle_type_chromosome.get(customer.data_index) is None:
                #     self.vehicle_type_chromosome[customer.data_index] = []
                # if vehicle_type not in self.vehicle_type_chromosome[customer.data_index]:
                #     self.vehicle_type_chromosome[customer.data_index].append(vehicle_type)

    def _reconstruct_chromosomes(self):
        """Constructs the giant tour and vehicle_index type chromosomes from the stored routes.
        Used after or during education."""
        for vehicle_type, tour in self.routes.items():
            self.giant_tour_chromosome[vehicle_type] = []
            for route in tour:
                self.giant_tour_chromosome[vehicle_type].extend(route.sequence)

        self._construct_vehicle_type_chromosome()

    def overwrite_with(self, other: "Individual"):
        """Overwrites this individual's attributes with the target individual's."""
        self.giant_tour_chromosome = other.giant_tour_chromosome
        self.vehicle_type_chromosome = other.vehicle_type_chromosome
        self.routes = other.routes
        self.cost = other.cost
        self.capacity_penalty = other.capacity_penalty
        self.duration_penalty = other.duration_penalty

    def evaluate(self, allow_completion: bool = True):
        """Splits giant tour, finds the cost and penalty for each route and sums for the entire individual"""
        # chrom_id = id(self.giant_tour_chromosome)
        # logger.debug(f"Eval giant tour: {chrom_id}, {self.giant_tour_chromosome}")

        if allow_completion:
            self.complete_customer_services_slow()

            if not self.verify_routes():
                self.split_all()

        # Retrieve some config variables and store them locally so that they don't have to be repeatedly retrieved
        allow_unlimited_fleets = run_settings.RUN_CONFIG.allow_unlimited_fleets
        waste_between_routes = run_settings.RUN_CONFIG.waste_between_routes
        daily_service_time = run_settings.RUN_CONFIG.daily_service_time

        self.cost = 0
        self.capacity_penalty = 0
        self.duration_penalty = 0
        for vehicle_type_index, routes in self.routes.items():
            vehicle_type = VehicleType(vehicle_type_index)
            # Also account for the cost of the number of vehicles used
            own_fleet_time = daily_service_time * vehicle_type.available_vehicles
            hours_of_service = 0

            # Account for the cost on the route itself
            for route in routes:
                # If there are enough available hours of service to serve this route,
                # and the route can be served by SPAR's fleet
                # Or if using limited fleets, the hours of service check doesn't matter
                if not allow_unlimited_fleets or (
                        (hours_of_service + route.total_time) <= own_fleet_time and not route.third_party_only):
                    # If the expected hours of service of SPAR's own fleet is sufficient, use base vehicle type costs
                    self.cost += route.cost
                    self.capacity_penalty += route.capacity_penalty
                    self.duration_penalty += route.duration_penalty
                    hours_of_service += route.total_time + waste_between_routes
                else:
                    # If the hours of service for this vehicle type is greater than SPAR's own fleet capacity,
                    # hire extra vehicles at a predetermined cost multiplier
                    self.cost += route.cost * vehicle_type.hired_cost_multiplier
                    self.capacity_penalty += route.capacity_penalty * vehicle_type.hired_cost_multiplier
                    self.duration_penalty += route.duration_penalty * vehicle_type.hired_cost_multiplier
                    # hours_of_service += route.total_time + waste_between_routes

            # self.hired_vehicles[vehicle_type_index] = max(math.ceil(
            #     (hours_of_service - own_fleet_time) / daily_service_time), 0)

    @property
    def penalty(self) -> float:
        if self.capacity_penalty is None or self.duration_penalty is None:
            self.evaluate()
        return self.capacity_penalty + self.duration_penalty

    def get_penalised_cost(self, extra_penalty: int = 1) -> float:
        if self.cost is None:
            self.evaluate()
        return self.cost + extra_penalty * self.penalty

    @property
    def feasible(self) -> bool:
        """Determine whether the solution is feasible"""
        # if self.penalty is None:
        #     self.evaluate()
        return self.penalty == 0
        # feasible = True
        # for route in self.routes:
        #     feasible = route.feasible
        #     if not feasible:
        #         break
        #
        # return feasible

    @property
    def capacity_feasible(self) -> bool:
        if self.capacity_penalty is None:
            self.evaluate()

        return self.capacity_penalty == 0

    @property
    def duration_feasible(self) -> bool:
        if self.duration_penalty is None:
            self.evaluate()

        return self.duration_penalty == 0

    @property
    def num_stops(self) -> int:
        """Returns a count of how many stops there are."""
        stops = 0
        for tour in self.giant_tour_chromosome.values():
            stops += len(tour)
        return stops

    def split_all(self, extra_penalty: int = 1):
        """Splits all the individual's giant tours into optimal routes."""
        # self.verify_giant_tour()

        # for vehicle_type_index in self.vehicle_type_chromosome:
        for vehicle_type_index, tour in self.giant_tour_chromosome.items():
            # Set the route to blank in case it has been split before
            self.routes[vehicle_type_index] = []
            vehicle_type = VehicleType(vehicle_type_index)

            # Find the demand of each customer that is satisfied in this period
            # demands = []
            # for customer in range(len(tour)):
            #     demands.append(self.giant_tour_chromosome[vehicle_type_index][customer].serviced_demand)

            # Separate the giant tour into individual routes
            if run_settings.RUN_CONFIG.allow_unlimited_fleets:
                split_routes = self.split_tour_unlimited(tour, vehicle_type, extra_penalty)
            else:
                split_routes = self.split_tour(tour, vehicle_type, extra_penalty)

            # logger.debug(f"Split routes {split_routes}")

            # Store the information about the routes created
            for route_index, route in enumerate(split_routes):
                # if len(route) == 0:
                #     continue
                # if isinstance(route, ArcLocation):
                #     route = numpy.array([route])

                self.routes[vehicle_type_index].append(Route(route, vehicle_type))
                # Record the indices for use in the education sequence
                for loc_index, location in enumerate(route):
                    # if location.data_index == 0:
                    #     raise ValueError("Depot in route!")

                    location.route_index = route_index
                    location.index_in_route = loc_index

        # logger.debug(f"Split giant tour {id(self.giant_tour_chromosome)} into routes\n{self.routes}")

        # if not self.verify_routes():
        #     raise ValueError(f"Routes don't match tour after split!\n{self.giant_tour_chromosome}\n{self.routes}")
        # self.verify_giant_tour()

    def complete_customer_services(self):
        """Look for any customers that aren't getting all the service they need, then make sure they get it.
        Aggressively tries to reduce the stops per customer, with the side-affect of giving preference to the
        first-encountered stops at each customer.

        Vidal's paper focuses on customers getting the visit frequency they require, but I will need to check that
        they get all their demand."""
        # First, make a list of every customer which can keep track of how much has been serviced across all tours
        check_customers: Dict[int, LocationChecker] = {}
        if not isinstance(self.possible_locations, numpy.ndarray):
            self.possible_locations = data_globals.ALL_CUSTOMERS

        for location in self.possible_locations:
            check_customers[location.data_index] = LocationChecker(location)

        vehicle_types = list(self.giant_tour_chromosome.keys())
        # vehicle_types = list(data_globals.ALL_VEHICLE_TYPES)
        # if len(vehicle_types) == 0:
        #     raise ValueError("No vehicle types in chromosome!")

        tour_changed = False
        deliveries_split = False

        # Go through all stops in the giant tour and set any stops that aren't delivering full demand or
        # vehicle_index to 0
        shuffle(vehicle_types)
        for vehicle_type in vehicle_types:
            pallet_capacity = run_settings.RUN_DATA.pallet_capacity[vehicle_type]
            # for customer in self.giant_tour_chromosome[vehicle_type]:
            stop_index = 0
            while stop_index < len(self.giant_tour_chromosome[vehicle_type]):
                stop = self.giant_tour_chromosome[vehicle_type][stop_index]

                current_customer = check_customers[stop.data_index]
                remaining_demand = current_customer.demand - current_customer.serviced_demand

                # Can't deliver more than a vehicle_index's capacity
                if stop.serviced_demand > pallet_capacity:
                    stop.serviced_demand = pallet_capacity
                    deliveries_split = True

                # Sometimes the crossover will lead to stores being over serviced
                # Only bother with the zeroing of demands if there is still demand that needs to be serviced,
                # otherwise remove unnecessary stops.
                if stop.serviced_demand > remaining_demand:
                    if remaining_demand > 0:
                        stop.serviced_demand = remaining_demand
                    else:
                        self.giant_tour_chromosome[vehicle_type].pop(stop_index)
                        tour_changed = True
                        continue
                elif not (stop.serviced_demand == current_customer.demand or stop.serviced_demand == pallet_capacity):
                    stop.serviced_demand = 0
                    deliveries_split = True

                current_customer.serviced_demand += stop.serviced_demand
                stop_index += 1

        # logger.debug(f"After reduction: {self.giant_tour_chromosome}")

        if deliveries_split:
            # Loop through all stops looking for any with 0 serviced demand. If there is remaining demand,
            # add as much service as possible. If there is not remaining demand, remove the customer.
            shuffle(vehicle_types)
            for vehicle_type in vehicle_types:
                pallet_capacity = run_settings.RUN_DATA.pallet_capacity[vehicle_type]
                stop_index = 0

                while stop_index < len(self.giant_tour_chromosome[vehicle_type]):
                    stop = self.giant_tour_chromosome[vehicle_type][stop_index]
                    current_customer = check_customers[stop.data_index]

                    # Only check zeroed stops
                    if stop.serviced_demand == 0:
                        # If there is no remaining demand for this customer, remove the customer
                        remaining_demand = current_customer.demand - current_customer.serviced_demand

                        # if remaining_demand < 0:
                        #     pass

                        if remaining_demand == 0:
                            self.giant_tour_chromosome[vehicle_type].pop(stop_index)
                            tour_changed = True
                            continue

                        # If there is still remaining demand, service as much of it as possible for the vehicle_index
                        # type
                        stop.serviced_demand = min(remaining_demand, pallet_capacity)
                        current_customer.serviced_demand += stop.serviced_demand
                    # elif current_customer.demand != current_customer.serviced_demand:
                    #     if customer.serviced_demand >= 0:
                    #         pass

                    stop_index += 1

            # if stop_index >= 0:
            #     pass

        # Loop through all the checkers and make sure that each customer is receiving enough service.
        # If not, then add new stops at the customer with random vehicles at random points in the tour.
        for customer_index, check_customer in check_customers.items():
            while check_customer.serviced_demand < check_customer.demand:
                vehicle_type = sample(vehicle_types, 1)[0]
                # If the unserviced demand is greater than the vehicle_index's capacity, it must be constrained.
                demand_to_service = min(run_settings.RUN_DATA.pallet_capacity[vehicle_type],
                                        check_customer.demand - check_customer.serviced_demand)
                # Create a new customer to be inserted
                new_stop = Location(data_index=customer_index, serviced_demand=demand_to_service,
                                    demand=check_customer.demand)
                # Add the customer in a random location in the giant tour
                self.giant_tour_chromosome[vehicle_type].insert(
                    math.floor(random() * len(self.giant_tour_chromosome[vehicle_type])), new_stop)
                # Also increment the customer's serviced demand appropriately
                check_customer.serviced_demand += new_stop.serviced_demand
                tour_changed = True

        # logger.debug(f"Comp giant tour: {self.giant_tour_chromosome}")

        if tour_changed:
            self._construct_vehicle_type_chromosome()

        # self.verify_giant_tour()

    def complete_customer_services_slow(self):
        """Version of complete_customer_services that less aggressively tries to reduce the number of stops per
        customer. This version of the procedure may be a bit slower, but will respect seeded solutions better."""
        # First, make a list of every customer which can keep track of how much has been serviced across all tours
        check_customers: Dict[int, LocationChecker] = {}
        if not isinstance(self.possible_locations, numpy.ndarray):
            self.possible_locations = data_globals.ALL_CUSTOMERS

        for location in self.possible_locations:
            check_customers[location.data_index] = LocationChecker(location)

        vehicle_types = list(self.giant_tour_chromosome.keys())
        # vehicle_types = list(data_globals.ALL_VEHICLE_TYPES)
        # if len(vehicle_types) == 0:
        #     raise ValueError("No vehicle types in chromosome!")

        tour_changed = False
        redistribute_service = True

        # Go through all stops in the giant tour and make sure that none deliver more than their vehicle's capacity.
        for vehicle_type_index in vehicle_types:
            pallet_capacity = run_settings.RUN_DATA.pallet_capacity[vehicle_type_index]
            stop_index = 0
            while stop_index < len(self.giant_tour_chromosome[vehicle_type_index]):
                stop = self.giant_tour_chromosome[vehicle_type_index][stop_index]
                current_customer = check_customers[stop.data_index]
                # remaining_demand = current_customer.demand - current_customer.serviced_demand

                # Can't deliver more than a vehicle's capacity
                if stop.serviced_demand == 0:
                    # print("Pop")
                    self.giant_tour_chromosome[vehicle_type_index].pop(stop_index)
                    tour_changed = True
                    continue
                elif stop.serviced_demand > pallet_capacity:
                    stop.serviced_demand = pallet_capacity
                elif stop.serviced_demand < pallet_capacity:
                    # Otherwise, sum the extra capacity in the vehicle for more pallets beyond what are in the load
                    current_customer.excess_capacity += pallet_capacity - stop.serviced_demand

                current_customer.serviced_demand += stop.serviced_demand
                stop_index += 1

        # logger.debug(f"After reduction: {self.giant_tour_chromosome}")
        demands_satisfied = [customer.satisfies_all_demand for customer in check_customers.values()]

        # If not all customer demands are satisfied, perform completion
        if not all(demands_satisfied):
            # Repeat this so long as stops whose demand could be redistributed could be found
            while redistribute_service:
                redistribute_service = False
                # Loop through and set any stops that aren't delivering full demand or vehicle capacity to 0
                shuffle(vehicle_types)
                for vehicle_type_index in vehicle_types:
                    pallet_capacity = run_settings.RUN_DATA.pallet_capacity[vehicle_type_index]
                    stop_index = 0
                    while stop_index < len(self.giant_tour_chromosome[vehicle_type_index]):
                        stop = self.giant_tour_chromosome[vehicle_type_index][stop_index]
                        current_customer = check_customers[stop.data_index]

                        # Find the excess capacity for other stops
                        other_excess_capacity = current_customer.excess_capacity - (pallet_capacity -
                                                                                    stop.serviced_demand)
                        # If there is enough capacity in other stops to handle the demand serviced in this stop,
                        # remove it
                        if other_excess_capacity > stop.serviced_demand:
                            current_customer.serviced_demand -= stop.serviced_demand
                            current_customer.excess_capacity -= pallet_capacity - stop.serviced_demand
                            self.giant_tour_chromosome[vehicle_type_index].pop(stop_index)
                            tour_changed = True
                            redistribute_service = True
                            continue

                        remaining_demand = current_customer.demand - current_customer.serviced_demand

                        if remaining_demand > 0:
                            # If the remaining demand is positive, increase the stop's service demand as much as
                            # possible
                            max_increase = min(remaining_demand, pallet_capacity - stop.serviced_demand)
                            current_customer.serviced_demand += max_increase
                            current_customer.excess_capacity -= max_increase
                            stop.serviced_demand += max_increase
                        elif remaining_demand < 0:
                            if remaining_demand + stop.serviced_demand > 0:
                                # Else, if the service at this stop can be reduced to eliminate the excess supply
                                # (without being reduced to zero), then do so
                                max_reduction = min(-remaining_demand, stop.serviced_demand)
                                current_customer.serviced_demand -= max_reduction
                                current_customer.excess_capacity += max_reduction
                                stop.serviced_demand -= max_reduction
                            else:
                                # Otherwise, remove the stop
                                current_customer.serviced_demand -= stop.serviced_demand
                                current_customer.excess_capacity -= pallet_capacity - stop.serviced_demand
                                self.giant_tour_chromosome[vehicle_type_index].pop(stop_index)
                                tour_changed = True
                                continue

                        stop_index += 1

            demands_satisfied = [customer.satisfies_all_demand for customer in check_customers.values()]

            # If still not all customer demand is satisfied, add new stops at random for the customers who still need
            # service
            if not all(demands_satisfied):
                for customer_index, check_customer in check_customers.items():
                    while check_customer.serviced_demand < check_customer.demand:
                        vehicle_type_index = sample(vehicle_types, 1)[0]
                        # If the unserviced demand is greater than the vehicle_index's capacity, it must be constrained.
                        demand_to_service = min(run_settings.RUN_DATA.pallet_capacity[vehicle_type_index],
                                                check_customer.demand - check_customer.serviced_demand)
                        # Create a new customer to be inserted
                        new_stop = Location(data_index=customer_index, serviced_demand=demand_to_service,
                                            demand=check_customer.demand)
                        # Add the customer in a random location in the giant tour
                        self.giant_tour_chromosome[vehicle_type_index].insert(
                            math.floor(random() * len(self.giant_tour_chromosome[vehicle_type_index])), new_stop)
                        # Also increment the customer's serviced demand appropriately
                        check_customer.serviced_demand += new_stop.serviced_demand
                        tour_changed = True

        # logger.debug(f"Comp giant tour: {self.giant_tour_chromosome}")

        if tour_changed:
            self._construct_vehicle_type_chromosome()

        # self.verify_giant_tour()

    def _regenerate_customer_stop_lists(self, check_customers: List[LocationChecker]):
        """Resets the check customer serviced demand and customer indices, then creates them again from the giant tour
        chromosome.

        This is done because each time a customer is removed from the tour, the index of every customer after that will
        decrease.

        This function completely regenerates the customer lists, rather than trying to update just the indices of
        stops after the popped customer, as that approach seemed to be accumulating errors."""
        for ch_cust in check_customers:
            ch_cust.serviced_demand = 0
            ch_cust.stop_type_indices = []

        for vehicle_type, tour in self.giant_tour_chromosome.items():
            for index, stop in enumerate(tour):
                # Try find the correct check_customer for this customer.
                for ch_cust in check_customers:
                    if ch_cust.data_index == stop.data_index:
                        check_customer = ch_cust
                        break
                else:
                    # If not found, then raise an exception - this shouldn't happen.
                    raise ValueError(f"Check customer not found for {stop}.")

                # Add information about this customer to the appropriate check customer
                # check_customer.stop_type_indices.append((vehicle_type, index))
                check_customer.serviced_demand += stop.serviced_demand

    # @staticmethod
    # def _update_customer_stop_lists(check_customers: List[LocationChecker], vehicle_type: int, index_to_pop: int):
    #     """If a certain customer is going to be removed from a tour, then the LocationChecker's vehicle_types and
    #     tour_indices lists need to be updated.
    #
    #     The LocationChecker correlating to the removed customer needs the customer
    #     removed, while individuals after the removed customer need their index updated.
    #
    #     This function updates the lists instead of entirely regenerating them."""
    #     # logger.debug(f"Updating customer lists {[customer.stop_type_indices for customer in check_customers]}")
    #     for customer in check_customers:
    #         to_pop = None
    #         for index in range(len(customer.stop_type_indices)):
    #             if customer.stop_type_indices[index][0] == vehicle_type:
    #                 if customer.stop_type_indices[index][1] == index_to_pop:
    #                     to_pop = index
    #                 elif customer.stop_type_indices[index][1] > index_to_pop:
    #                     customer.stop_type_indices[index] = (customer.stop_type_indices[index][0],
    #                                                          customer.stop_type_indices[index][1] - 1)
    #         # logger.debug(f"Popping {to_pop} in {customer.stop_type_indices}")
    #         if to_pop:
    #             customer.stop_type_indices.pop(to_pop)
    #     # logger.debug(f"Updated customer lists {[customer.stop_type_indices for customer in check_customers]}")

    @staticmethod
    def split_tour(tour: List[Location], vehicle_type: VehicleType, extra_penalty: int) -> List[List[Location]]:
        """Split the given giant tour into individual routes using the algorithm from "Technical note: Split algorithm
        in O(n) for the capacitated vehicle_type routing problem" (Vidal, et al. 2015).

        Uses Algorithm 3 (limited vehicles) with the soft capacity constraints extension and the soft duration limit
        from jjupe's implementation."""

        # Define helper functions
        def dominates(i: int, j: int, k: int):
            """Test if t dominates j as a predecessor for all nodes x >= j+1"""
            left = potential[k][i] + depot_tour[i + 1].travel_times[depot.data_index] - sum_duration[i + 1]
            right = potential[k][j] + depot_tour[j + 1].travel_times[depot.data_index] - sum_duration[j + 1]
            if i <= j:
                return left + extra_penalty * (sum_load[j] - sum_load[i]) <= right
            else:
                return left <= right

        def cost(i: int, j: int):
            """If the load is not more than 2 times the vehicle_type capacity,
            return the cost of travelling to the next step.

            The cost includes the unload time and any penalties."""
            if sum_load[j] - sum_load[i] < 2 * vehicle_capacity:
                return depot_tour[i + 1].travel_times[depot.data_index] + sum_duration[j] - \
                       sum_duration[i + 1] + depot_tour[j].travel_times[depot.data_index] + extra_penalty * \
                       (run_settings.RUN_CONFIG.capacity_penalty_multiplier *
                        max(sum_load[j] - sum_load[i] - vehicle_capacity, 0) +
                        run_settings.RUN_CONFIG.duration_penalty_multiplier *
                        max(sum_duration[j] - sum_duration[i + 1] - run_settings.RUN_CONFIG.daily_drive_limit, 0))
            else:
                return math.inf
            # return sum_duration[j] - sum_duration[i + 1] + depot_tour[i + 1].travel_times[0] + \
            #        depot_tour[j].travel_times[0]

        # logger.debug(f"Splitting {tour} for vehicle_index type {vehicle_type}")
        if len(tour) <= 1:
            return [tour]

        vehicle_capacity = vehicle_type.capacity
        num_vehicles = vehicle_type.available_vehicles

        # distance_cost = run_settings.RUN_DATA.distance_cost[vehicle_type]
        # time_cost = run_settings.RUN_DATA.time_cost[vehicle_type]

        # The first element of the tour indicates which location to use as the depot.
        # This is not needed for my usage of the algorithm, but it seems baked in to the split algorithm
        depot = data_globals.DEPOT
        depot_tour = numpy.array([depot, *tour])

        num_stops = len(depot_tour)

        # DON'T USE: Python treats the inner lists as the same object, meaning changes to one affect the others
        # potential = [[math.inf] * num_stops] * (num_vehicles + 1)

        potential = numpy.array([numpy.array([math.inf for _ in range(num_stops)]) for _ in range(num_vehicles + 1)])
        potential[0][0] = 0

        # predecessor = [[0] * num_stops] * (num_vehicles + 1)
        predecessor = numpy.array([numpy.array([0 for _ in range(num_stops)]) for _ in range(num_vehicles + 1)])
        sum_load = numpy.array([0] * num_stops)
        sum_duration = numpy.array([0] * num_stops)

        for t in range(num_stops):
            # Find the current and previous stops
            current_stop = depot_tour[t]
            # prev_stop: ArcLocation = None
            if t > 0:
                prev_stop: Location = depot_tour[t - 1]
                sum_load[t] = sum_load[t - 1]
                sum_duration[t] = sum_duration[t - 1]
            else:
                continue
            # Add the demand of the customer to the load sum
            sum_load[t] += current_stop.serviced_demand
            # Add the expected service duration (demand * average pallet unload) to the duration sum
            # Add the expected travel time
            sum_duration[t] += prev_stop.travel_times[current_stop] + current_stop.expected_unload_time

        for k in range(num_vehicles):
            # Reset the queue to begin with the current vehicle_type index
            queue = [k]
            # print("Out", queue)
            # Loop through stores, but start at the index of the current vehicle_type
            for t in range(k + 1, num_stops):
                if len(queue) == 0:
                    break

                # print("In", queue)
                # print("predecessor ", predecessor)
                # print("potential ", potential)
                # The front of the queue is the best predecessor for t
                potential[k + 1][t] = potential[k][queue[0]] + cost(queue[0], t)
                predecessor[k + 1][t] = queue[0]

                if t < num_stops - 1:
                    # If t is not dominated by the last pile
                    if not dominates(queue[-1], t, k):
                        # then t will be inserted, after removing each item that it dominates
                        while len(queue) > 0 and dominates(t, queue[-1], k):
                            queue.pop()
                        queue.append(t)

                    # Check if the front is able to reach the next node, otherwise pop it.
                    while len(queue) > 1 and potential[k][queue[0]] + cost(queue[0], t + 1) >= \
                            potential[k][queue[1]] + cost(queue[1], t + 1):
                        # while len(queue) > 1 and sum_load[t + 1] - sum_load[queue[0]] > vehicle_capacity:
                        queue.pop(0)

                # The following three paragraphs are adapted from Vidal's code
        # Find the optimal number of routes
        min_cost = math.inf
        opt_num_vehicles = 0
        for k in range(1, num_vehicles + 1):
            if potential[k][num_stops - 1] < min_cost:
                min_cost = potential[k][num_stops - 1]
                opt_num_vehicles = k

        # logger.debug("predecessor ", predecessor)
        # logger.debug("potential ", potential)
        # logger.debug("vehicle_capacity ", vehicle_capacity, "opt_num_vehicles ", opt_num_vehicles, "num_stops",
        #              num_stops)
        # logger.debug("sum_load ", sum_load)

        # Check that the algorithm managed to find a routes
        if min_cost == math.inf:
            # logger.warning("No split routes propagated to the last node")
            # Need to cut out the first element, because this is the depot
            return [tour]

        # # Fill the routes run_data in the optimal order
        # cour = num_stops - 1
        # routes = [0] * opt_num_vehicles
        # for i in range(opt_num_vehicles - 1, -1, -1):
        #     cour = predecessor[i + 1][cour]
        #     routes[i] = cour + 1
        #
        # print("routes ", routes)

        # Adapted from jjupe's code
        # Compile the routes
        start = predecessor[opt_num_vehicles][num_stops - 1]
        routes = [[depot_tour[x] for x in range(start + 1, num_stops)]]
        index = start
        for i in range(opt_num_vehicles - 1, 0, -1):
            start = predecessor[i][index]
            routes.append([depot_tour[x] for x in range(start + 1, index + 1)])
            index = start

        routes = list(reversed(routes))

        # logger.debug(f"Routes {routes}")
        # Verify that routes match the tour
        # tour_indices = [stop.data_index for stop in depot_tour[1:]]
        # route_indices = [stop.data_index for route in routes for stop in route]
        # if not Counter(tour_indices) == Counter(route_indices):
        #     raise ValueError("Routes don't match tour!")

        # for route in routes:
        #     for stop in route:
        #         if stop.data_index == 0:
        #             raise ValueError("Depot in route!")

        # logger.debug(f"Split routes {routes}")

        return routes

    @staticmethod
    def split_tour_unlimited(tour: List[Location], vehicle_type: VehicleType, extra_penalty: int) -> List[
        List[Location]]:
        """Split the given giant tour into individual routes using the algorithm from "Technical note: Split algorithm
        in O(n) for the capacitated vehicle_type routing problem" (Vidal, et al. 2015).

        Uses Algorithm 2 (unlimited vehicles)."""

        # Define helper functions
        def dominates(i: int, j: int):
            """Test if t dominates j as a predecessor for all nodes x >= j+1"""
            left = potential[i] + depot_tour[i + 1].travel_distances[data_globals.DEPOT] - sum_distances[i + 1]
            right = potential[j] + depot_tour[j + 1].travel_distances[data_globals.DEPOT] - sum_distances[j + 1]
            if i <= j:
                return sum_load[i] == sum_load[j] and left <= right
            else:
                return left <= right

        def cost(i: int, j: int):
            """If the load is not more than 2 times the vehicle_type capacity,
            return the cost of travelling to the next step.

            The cost includes the unload time and any penalties."""
            if sum_load[j] - sum_load[i] <= vehicle_capacity:
                return depot_tour[i + 1].travel_distances[depot.data_index] + sum_distances[j] - \
                       sum_distances[i + 1] + depot_tour[j].travel_distances[depot.data_index]
            else:
                return math.inf
            # return sum_distances[j] - sum_distances[i + 1] + depot_tour[i + 1].travel_times[depot.data_index] + \
            #        depot_tour[j].travel_times[depot.data_index]

        # logger.debug(f"Splitting {tour} for vehicle_index type {vehicle_type}")
        if len(tour) <= 1:
            return [tour]

        vehicle_capacity = vehicle_type.capacity

        # The first element of the tour indicates which location to use as the depot.
        # This is not needed for my usage of the algorithm, but it seems baked in to the split algorithm
        depot = data_globals.DEPOT
        depot_tour = [depot, *tour]

        num_stops = len(depot_tour)

        potential = [math.inf for _ in range(num_stops)]
        potential[0] = 0

        predecessor = [0 for _ in range(num_stops)]
        sum_load = [0] * num_stops
        sum_distances = [0] * num_stops

        for t in range(num_stops):
            # Find the current and previous stops
            current_stop = depot_tour[t]
            # prev_stop: ArcLocation = None
            if t > 0:
                prev_stop: Location = depot_tour[t - 1]
                sum_load[t] = sum_load[t - 1]
                sum_distances[t] = sum_distances[t - 1]
            else:
                continue
            # Add the demand of the customer to the load sum
            sum_load[t] += current_stop.serviced_demand
            # Add the expected service duration (demand * average pallet unload) to the duration sum
            # Add the expected travel time
            sum_distances[t] += prev_stop.travel_distances[current_stop]  # + current_stop.expected_unload_time

        queue = [0]
        # Loop through stores, but start at the index of the current vehicle_type
        for t in range(1, num_stops):
            if len(queue) == 0:
                break

            # The front of the queue is the best predecessor for t
            potential[t] = potential[queue[0]] + cost(queue[0], t)
            predecessor[t] = queue[0]

            if t < num_stops - 1:
                # If t is not dominated by the last pile
                if not dominates(queue[-1], t):
                    # then t will be inserted, after removing each item that it dominates
                    while len(queue) > 0 and dominates(t, queue[-1]):
                        queue.pop()
                    queue.append(t)

                # Check if the front is able to reach the next node, otherwise pop it.
                # while len(queue) > 1 and potential[queue[0]] + cost(queue[0], t + 1) < math.inf:
                # while len(queue) > 1 and potential[queue[0]] + cost(queue[0], t + 1) >= potential[queue[1]] + \
                #         cost(queue[1], t + 1):
                while len(queue) > 1 and sum_load[t + 1] - sum_load[queue[0]] > vehicle_capacity:
                    queue.pop(0)

        # Check that the algorithm managed to find a routes
        if potential[-1] == math.inf:
            return [tour]

        # Find the number of routes used
        opt_num_vehicles = 0
        index = num_stops - 1
        while index != 0:
            index = predecessor[index]
            opt_num_vehicles += 1

        # Adapted from jjupe's code
        # Compile the routes
        start = predecessor[num_stops - 1]
        routes = [[depot_tour[x] for x in range(start + 1, num_stops)]]
        index = start
        for i in range(opt_num_vehicles - 1, 0, -1):
            start = predecessor[index]
            routes.append([depot_tour[x] for x in range(start + 1, index + 1)])
            index = start

        routes = list(reversed(routes))

        # logger.debug(f"Routes {routes}")
        # Verify that routes match the tour
        # tour_indices = [stop.data_index for stop in depot_tour[1:]]
        # route_indices = [stop.data_index for route in routes for stop in route]
        # if not Counter(tour_indices) == Counter(route_indices):
        #     raise ValueError("Routes don't match tour!")

        # for route in routes:
        #     for stop in route:
        #         if stop.data_index == 0:
        #             raise ValueError("Depot in route!")

        # logger.debug(f"Split routes {routes}")

        return routes

    def educate(self, extra_penalty: int = 1):
        """A local search is applied to the individual based on Vidal et al. (2011)'s paper.
        There are two local search procedures: route improvement (RI) and pattern improvement (PI).
        Education applies RI, PI, and then RI again.

        jjupe's code was also used as reference for aspects of the algorithm."""
        # self.verify_giant_tour()
        # Before starting, check that the existing routes are valid for the giant tour chromosome
        # logger.debug(f"Giant tour before education: {self.giant_tour_chromosome}")
        if not self.verify_routes():
            self.split_all()
        # logger.debug(f"Routes before education: {self.routes}")
        self._route_improvement(extra_penalty)
        self._pattern_improvement(extra_penalty)
        self._route_improvement(extra_penalty)
        self.evaluate()

        # self.verify_giant_tour()

    def verify_routes(self) -> bool:
        """Check that the stored routes match the giant tour chromosome."""
        if self.routes is None:
            return False

        for vehicle_type, tour in self.giant_tour_chromosome.items():
            tour_indices = [stop.data_index for stop in tour]
            route_indices = [stop.data_index for route in self.routes[vehicle_type] for stop in route.sequence]
            if not Counter(tour_indices) == Counter(route_indices):
                return False
        return True

    def verify_giant_tour(self):
        """Check that the giant tour doesn't have any depot visits in it. 
        This function was added to help with with finding a bug that was causing depot visits to randomly appear in the 
        tour."""
        for vehicle_type, tour in self.giant_tour_chromosome.items():
            tour_indices = [stop.data_index for stop in tour]

            if tour_indices.count(0) > 0:
                raise ValueError("Depot in route!")

    def _route_improvement(self, extra_penalty: int):
        """Perform route improvement local search procedure.

        This procedure seeks to improve generated routes by swapping individual stops or sequences of stops visits
        between routes of the same vehicle_index type."""
        # self.verify_giant_tour()
        # Loop through each tour within the giant tour
        start_time = time()

        for vehicle_type_index, tour in self.routes.items():
            # Make sure all stops in the tour have the correct indices stored
            self._update_routes_location_indices(tour)
            vehicle_type = VehicleType(vehicle_type_index)

            for route in tour:
                # For each vertex, see if any improvements can be found
                # Don't iterate over the last, because a successor needs to be found each time
                for vertex in route.sequence[:-1]:
                    # Don't go over max time
                    if time() - start_time > run_settings.RUN_CONFIG.procedure_time_limit:
                        break
                    # Random neighborhood size
                    neighborhood_size = min(int(run_settings.RUN_CONFIG.granularity_threshold * len(tour)),
                                            vehicle_type.available_vehicles)
                    # Sort the list of distances to other locations in the tour
                    # sorted_distances = sorted(
                    #     [(vertex.find_correlation(x, vehicle_type), x) for x in route.sequence if x is not vertex])
                    sorted_distances = sorted(
                        [(vertex.find_correlation(x, vehicle_type), x) for x in route.sequence if x is not vertex],
                        key=lambda x: x[0])
                    if len(sorted_distances) > neighborhood_size:
                        sorted_distances = sorted_distances[0:neighborhood_size]
                    neighborhood = [stop_tuple[1] for stop_tuple in sorted_distances]

                    # Shuffle the neighborhood to explore it in a random order
                    shuffle(neighborhood)
                    # To improve the education, the best move is only applied when 5% of the neighborhood has been
                    # explored since the last one
                    improvement_gap = neighborhood_size * run_settings.RUN_CONFIG.neighborhood_improvement_gap
                    improvement_counter = 0
                    # The possible moves that can be tried
                    possible_moves = [self._one_insertion, self._two_insertion, self._two_swap_insertion,
                                      self._one_swap, self._two_swap, self._double_swap, self._two_opt,
                                      self._two_opt_star, self._two_opt_star_reverse]
                    # Store the results of the best move
                    # best_move: List[ArcRoute] = self.routes[vehicle_type]
                    best_move: List[Route] = tour
                    best_move_cost: float = self._sum_routes_cost(best_move)
                    current_cost: float = best_move_cost
                    # Now iterate over the neighbors of the vertex
                    # Don't iterate over the last, because a successor needs to be found each time
                    for index, neighbor in enumerate(neighborhood[:-1]):
                        # These don't need to be defined here, but I'm doing so so that my IDE stops warning me they
                        # might not be defined in the checks after the loop
                        new_cost = best_move_cost
                        tour_copy = best_move

                        # logger.debug(f"vertex: {vertex}, neighbor: {neighbor}")
                        # logger.debug(f"tour: {tour}")

                        # test = False
                        # Make sure that neither vertex nor their successors are the same
                        if vertex.route_index == neighbor.route_index:
                            if abs(vertex.index_in_route - neighbor.index_in_route) <= 1:
                                # logger.error("Neighbor too close to vertex")
                                # test = True
                                continue

                        # Make sure that neither the vertex nor neighbor are the last customer in the sequence
                        if not vertex.index_in_route < len(tour_copy[vertex.route_index].sequence) - 1:
                            # logger.error("Vertex is last customer")
                            # test = True
                            continue

                        if not neighbor.index_in_route < len(tour_copy[neighbor.route_index].sequence) - 1:
                            # logger.error("Neighbor is last customer")
                            # test = True
                            continue
                        # if test:
                        #     logger.error("Continue didn't work")
                        #     raise ValueError("Continue didn't work.")

                        # The moves are examined in a random order.
                        # The first move to yield an improvement will be implemented.
                        # If none of the moves made an improvement, cease the search for improvements on this vertex.
                        shuffle(possible_moves)
                        improved = False
                        try:
                            for move in possible_moves:
                                # logger.debug(f"tour: {tour}")
                                # logger.debug(f"tour_copy: {tour_copy}")
                                # Don't go over max time
                                if time() - start_time > run_settings.RUN_CONFIG.procedure_time_limit:
                                    break

                                tour_copy = deepcopy(self.routes[vehicle_type.data_index])
                                old_cost = self._sum_routes_cost(tour_copy)

                                # Find the vertices needed for any possible move
                                vertex_copy = tour_copy[vertex.route_index].sequence[vertex.index_in_route]
                                vertex_successor = tour_copy[vertex.route_index].sequence[
                                    vertex.index_in_route + 1]
                                neighbor_copy = tour_copy[neighbor.route_index].sequence[neighbor.index_in_route]
                                neighbor_successor = tour_copy[neighbor.route_index].sequence[
                                    neighbor.index_in_route + 1]

                                # Make the move
                                u_route_index = vertex_copy.route_index
                                v_route_index = neighbor_copy.route_index
                                # logger.debug(f"tour_copy[u_route_index]: {tour_copy[u_route_index]},\n"
                                #              f"tour_copy[v_route_index]: {tour_copy[v_route_index]}")
                                u_route = tour_copy[u_route_index]
                                v_route = tour_copy[v_route_index]
                                u_sequence = list(u_route.sequence)
                                v_sequence = list(v_route.sequence)
                                # logger.debug(f"u_route: {u_route},\nv_route: {v_route}")
                                move(u_sequence=u_sequence, v_sequence=v_sequence, u=vertex_copy,
                                     v=neighbor_copy, x=vertex_successor, y=neighbor_successor)

                                u_route.sequence = numpy.array(u_sequence)
                                v_route.sequence = numpy.array(v_sequence)

                                # Update the route index and index in route for both routes
                                self._update_location_indices(u_route, u_route_index)
                                self._update_location_indices(v_route, v_route_index)

                                # Evaluate the routes
                                u_route.evaluate(extra_penalty)
                                v_route.evaluate(extra_penalty)

                                # See whether the move was an improvement
                                new_cost = self._sum_routes_cost(tour_copy)

                                if new_cost < old_cost:
                                    improved = True
                                    break
                        except IndexError:
                            continue

                        # Check if an improvement was made
                        if improved:
                            if new_cost < best_move_cost:
                                best_move = tour_copy
                        else:
                            # ArcStop iterating over this neighborhood if no improving moves were found for this neighbor
                            break

                        # Check if 5% of the neighborhood has been explored since last improving move
                        if index > improvement_gap * (improvement_counter + 1):
                            if best_move_cost < current_cost:
                                self._update_routes_location_indices(best_move)
                                self.routes[vehicle_type.data_index] = best_move
                                improvement_counter += 1
                            else:
                                # If no improving move was found, then customer iterating over this vertex
                                break

        self._reconstruct_chromosomes()
        # self.verify_giant_tour()

    @staticmethod
    def _sum_routes_cost(routes: List[Route]) -> float:
        """Sums the costs and penalties of a list of routes."""
        cost = 0
        for route in routes:
            cost += route.cost + route.penalty
        return cost

    def update_all_routes_location_indices(self):
        """Updates the route location indices for every vehicle_index's routes."""
        for vehicle_type, routes in self.routes.items():
            self._update_routes_location_indices(routes)

    @staticmethod
    def _update_routes_location_indices(routes: List[Route]):
        """Updates the route_index and index_in_route that each location stores in every route in the list."""
        # logger.debug(f"Updating location indices: {routes}")
        for route_index, route in enumerate(routes):
            Individual._update_location_indices(route, route_index)
            # logger.debug(f"route_index: {route_index}, route: {route}")
        # logger.debug(f"Updated location indices: {routes}")

    @staticmethod
    def _update_location_indices(route: Route, route_index: int):
        """Updates the route_index and index_in_route that each location stores for the given route."""
        for stop_index, location in enumerate(route.sequence):
            location.route_index = route_index
            location.index_in_route = stop_index

    @staticmethod
    def _one_insertion(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location, x: Location,
                       y: Location):
        """M1 of Vidal et al. (2011).
        Remove u and place it after v."""
        u_sequence.remove(u)
        v_sequence.insert(v.index_in_route + 1, u)

    @staticmethod
    def _two_insertion(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location, x: Location,
                       y: Location):
        """M2 of Vidal et al. (2011).
        Remove u and x, and place them after v."""
        u_sequence.remove(u)
        u_sequence.remove(x)
        v_sequence.insert(v.index_in_route + 1, x)
        v_sequence.insert(v.index_in_route + 1, u)

    @staticmethod
    def _two_swap_insertion(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location,
                            x: Location, y: Location):
        """M3 of Vidal et al. (2011).
        Remove u and x, swap them, and then place them after v."""
        Individual._two_insertion(u_sequence=u_sequence, v_sequence=v_sequence, u=x, v=v, x=u, y=y)

    @staticmethod
    def _one_swap(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location, x: Location,
                  y: Location):
        """M4 of Vidal et al. (2011).
        Swap u and v."""
        u_sequence.remove(u)
        v_sequence.remove(v)
        u_sequence.insert(u.index_in_route + 1, v)
        v_sequence.insert(v.index_in_route + 1, u)

    @staticmethod
    def _two_swap(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location, x: Location,
                  y: Location):
        """M5 of Vidal et al. (2011).
        Swap u and x with v."""
        u_sequence.remove(u)
        u_sequence.remove(x)
        v_sequence.remove(v)
        u_sequence.insert(u.index_in_route + 1, v)
        v_sequence.insert(v.index_in_route + 1, x)
        v_sequence.insert(v.index_in_route + 1, u)

    @staticmethod
    def _double_swap(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location, x: Location,
                     y: Location):
        """M6 of Vidal et al. (2011).
        Swap u and x with v and y."""
        u_sequence.remove(u)
        u_sequence.remove(x)
        v_sequence.remove(v)
        v_sequence.remove(y)
        u_sequence.insert(u.index_in_route + 1, y)
        u_sequence.insert(u.index_in_route + 1, v)
        v_sequence.insert(v.index_in_route + 1, x)
        v_sequence.insert(v.index_in_route + 1, u)

    @staticmethod
    def _two_opt(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location, x: Location,
                 y: Location):
        """In place of M7 of Vidal et al. (2011).
        If u and v are in the same route, replace sequence from u-x and v-y with u-y and v-x.
        This reverses the route from x-y.

        Vidal et al. (2011)'s paper says replace u-x and v-y with u-v and x-y, which would separate the route into two.
        This is not in line with the standard concept of the 2-opt move.
        Their 2012 paper says to reverse the route between two locations, which matches with the standard 2-opt."""
        if u_sequence is v_sequence:
            reverse_start, reverse_end = x.index_in_route, y.index_in_route
            if reverse_start > reverse_end:
                reverse_start, reverse_end = reverse_end, reverse_start
            u_sequence[reverse_start:reverse_end] = reversed(u_sequence[reverse_start:reverse_end])

    @staticmethod
    def _two_opt_star(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location, x: Location,
                      y: Location):
        """M9 of Vidal et al. (2011).
        If u and v are in different routes, replace sequence from u-x and v-y with u-y and x-v.
        This will swap the ends of two routes."""
        if u_sequence is not v_sequence:
            # Popping the last element of the sequence produces a reversed list, so it needs to be reversed again
            u_end = reversed([u_sequence.pop() for _ in range(x.index_in_route, len(u_sequence))])
            v_end = reversed([v_sequence.pop() for _ in range(y.index_in_route, len(v_sequence))])
            u_sequence.extend(v_end)
            v_sequence.extend(u_end)

    @staticmethod
    def _two_opt_star_reverse(u_sequence: List[Location], v_sequence: List[Location], u: Location, v: Location,
                              x: Location, y: Location):
        """M8 of Vidal et al. (2011).
        If u and v are in different routes, replace sequence from u-x and v-y with u-v and x-y.
        This will join the starts of both routes into a new route and the ends of them into another.
        Neither sequence's start or end is reversed, as opposed to what Vidal's notation implies."""
        if u_sequence is not v_sequence:
            u_end = reversed([u_sequence.pop() for _ in range(x.index_in_route, len(u_sequence))])
            v_start = [v_sequence.pop(0) for _ in range(y.index_in_route)]
            u_sequence.extend(v_start)
            v_sequence.extend(u_end)

    def _pattern_improvement(self, extra_penalty: int):
        """Perform pattern improvement local search procedure.

        My implementation likely differs somewhat from what Vidal had in mind, as his explanation was too brief.

        This procedure seeks to improve the order (pattern) of stops in tours of any vehicle_index type. For each customer in
        random order, find the difference in cost for removing it from its current location and inserting it into every
        other possible location. This move cost is calculated merely as the change in travel distance and time
        multiplied by the appropriate vehicle_index's costs.

        The top 5% of all possibly insertion points will be stored. Once all insertion points have been explored, then
        select a random one from the top 5% to execute a split on. If the generated routes have a lower total cost than
        the current routes, the move is executed.

        If a move is executed, the loop through all stops is broken, the customer list is reshuffled and the loop is
        initiated again.

        If all stops are checked without execution, or the run time limit is hit, then the pattern improvement procedure
        concludes."""
        # self.verify_giant_tour()
        # for vehicle_type, tour in self.routes.items():
        #     tour_stops = enumerate([customer for route in tour for customer in route.sequence])
        # for vehicle_type, tour in self.giant_tour_chromosome.items():
        #     tour_stops = tour.copy()
        #     broken = True
        #     # If a new spot to place the customer in is found, then the all stops will need to be checked again
        #     while broken:
        #         broken = False
        #         shuffle(tour_stops)
        #         # Check each customer in random order
        #         for customer in tour_stops:
        #             stop_route = self.routes[vehicle_type][customer.route_index]
        #             removal_reduction = stop_route.get_removal_reduction(customer.index_in_route, extra_penalty)
        #             insertion_costs = []
        #             cheapest_location = (-1, -1)
        #             cheapest_cost = removal_reduction
        #             # Check each possible location to insert the customer
        #             for route in self.routes[vehicle_type]:
        #                 insertion_costs.append([])
        #                 for insertion_index in range(len(route.sequence) + 1):
        #                     insert_cost = route.get_insertion_cost(customer, insertion_index, extra_penalty)
        #                     insertion_costs[insertion_index].append(insert_cost)
        #                     if insert_cost < cheapest_cost:
        #                         cheapest_cost = insert_cost
        #                         cheapest_location = (insertion_index, len(insertion_costs[insertion_index]) - 1)
        broken = True

        start_time = time()
        while broken and time() - start_time < run_settings.RUN_CONFIG.procedure_time_limit:
            # Create a list of all the stops in the giant tour.
            # The list shall be recreated each time a move is preformed, as this is easier than looping through the
            # whole list and updating the indices of any stops that are after the removal or insertion points
            # print(f"{sum([len(tour) for tour in self.giant_tour_chromosome.values()])}")
            all_stops: List[Tuple[int, int, Location]] = [(vehicle_type, index, stop) for vehicle_type, tour in
                                                          self.giant_tour_chromosome.items() for index, stop in
                                                          enumerate(tour)]

            best_insertions: List[Tuple[float, int, int]] = []
            broken = False
            shuffle(all_stops)
            # Go through each of the stops in the giant tour in random order.
            for stop in all_stops:
                remove_vehicle_type = stop[0]
                remove_index = stop[1]
                remove_stop = stop[2]

                # Estimate cost of removing this customer
                if remove_index > 0:
                    predecessor = self.giant_tour_chromosome[remove_vehicle_type][remove_index - 1]
                else:
                    predecessor = data_globals.DEPOT
                if remove_index < len(self.giant_tour_chromosome[remove_vehicle_type]) - 1:
                    successor = self.giant_tour_chromosome[remove_vehicle_type][remove_index + 1]
                else:
                    successor = data_globals.DEPOT

                remove_cost_reduction = self._estimate_move_cost(remove_vehicle_type, predecessor, remove_stop,
                                                                 successor)

                # Look through all possible insertion positions
                for vehicle_type, tour in self.giant_tour_chromosome.items():
                    # Loop past the end of each tour, as the customer will be inserted before the index
                    for insertion_index in range(len(tour)):
                        # Estimate the cost of inserting the customer
                        if insertion_index > 0:
                            predecessor = tour[insertion_index - 1]
                        else:
                            predecessor = data_globals.DEPOT
                        if insertion_index < len(tour):
                            successor = tour[insertion_index]
                        else:
                            successor = data_globals.DEPOT

                        insertion_cost = self._estimate_move_cost(vehicle_type, predecessor, remove_stop, successor)

                        # If the insertion cost is less than the remove cost reduction,
                        # try add it to the best insertions list
                        if insertion_cost < remove_cost_reduction:
                            # If the list is longer than 5% of the total number of stops,
                            # then replace the worst in the list if this one is better.
                            if len(best_insertions) >= len(
                                    all_stops) * run_settings.RUN_CONFIG.neighborhood_improvement_gap:
                                best_insertions = sorted(best_insertions)
                                if insertion_cost < best_insertions[-1][0]:
                                    best_insertions.pop()
                                    best_insertions.append((insertion_cost, vehicle_type, insertion_index))
                            else:
                                # Otherwise, simply add it to the list
                                best_insertions.append((insertion_cost, vehicle_type, insertion_index))

                # Now select a random insertion from the list of best insertions,
                # then split the tours and compare their costs
                if len(best_insertions) > 0:
                    selected_move: Tuple[float, int, int] = sample(best_insertions, 1)[0]
                    insert_vehicle_type = selected_move[1]
                    insertion_index = selected_move[2]

                    # Make the change to the giant tour and create a new individual to take advantage of the solution
                    # function. This is less computationally efficient than evaluating for only the changed tours, but
                    # it also allows the usage of the complete_customer_services function.
                    potential_giant_tour = deepcopy(self.giant_tour_chromosome)
                    potential_giant_tour[insert_vehicle_type].insert(insertion_index,
                                                                     potential_giant_tour[remove_vehicle_type].pop(
                                                                         remove_index))

                    potential_individual = Individual(potential_giant_tour,
                                                      possible_locations=self.possible_locations)
                    # potential_individual.complete_customer_services()
                    # potential_individual.evaluate(extra_penalty)
                    potential_individual.evaluate()

                    # If an improving move was made, then break out of the loop through all stops and restart it
                    if potential_individual.get_penalised_cost(extra_penalty) < self.get_penalised_cost(
                            extra_penalty):
                        self.overwrite_with(potential_individual)
                        broken = True
                        break
                # If no improving move was made, the procedure will loop to the next customer in the randomised list

        # if broken:
        #     logger.debug("Pattern Improvement time limit hit")
        # else:
        #     logger.debug("Pattern Improvement completed")
        # self.verify_giant_tour()

    @staticmethod
    def _estimate_move_cost(vehicle_type: int, predecessor: Location, stop: Location, successor: Location) -> float:
        return run_settings.RUN_DATA.distance_cost[vehicle_type] * \
               (predecessor.travel_distances[stop.data_index] + stop.travel_distances[successor.data_index] -
                predecessor.travel_distances[successor.data_index]) + \
               run_settings.RUN_DATA.time_cost[vehicle_type] * \
               (predecessor.travel_times[stop.data_index] + stop.travel_times[successor.data_index] -
                predecessor.travel_times[successor.data_index])

# class IndividualProxy(BaseProxy, Individual):
#     """A proxy object that allows information on and individual to be shared between processes"""
#     def __init__(self, token: Any, serializer: str, giant_tour_chromosome, vehicle_type_chromosome, possible_locations):
#         super(BaseProxy).__init__(token, serializer)
#         super(Individual).__init__(giant_tour_chromosome, vehicle_type_chromosome, possible_locations)
