from copy import copy
from math import ceil
from typing import List, Optional, Tuple

from numpy import array, ndarray

from model import run_settings
from util import create_logger

logger = create_logger(__name__)


class Location:
    """Stores a location index, ID and serviced demand.
    This class is used so that a separate list to track demand per customer is not needed."""

    def __init__(self, data_index: int, serviced_demand: int = None, demand: int = None):
        # The index of this location in run_data
        self.data_index: int = data_index

        # These values will be assigned by the split algorithm and are used by the education
        self.route_index: Optional[int] = None
        self.index_in_route: Optional[int] = None
        self.name: str = run_settings.RUN_DATA.locations[self.data_index]
        self.actual_demand: int = ceil(run_settings.RUN_DATA.demand[self.data_index])
        self.window_start: float = run_settings.RUN_DATA.window_start[self.data_index]
        self.window_end: float = run_settings.RUN_DATA.window_end[self.data_index]
        self.travel_distances: Tuple[float, ...] = run_settings.RUN_DATA.distances[self.data_index]
        self.travel_times: Tuple[float, ...] = run_settings.RUN_DATA.times[self.data_index]

        # For use in decomposed problems. Allows the actual_demand value to be overridden.
        self._demand: Optional[int] = None
        if demand:
            self._demand = demand
        # The number of pallets being delivered to this instance of this location
        if serviced_demand is not None:
            self.serviced_demand: int = serviced_demand
        else:
            self.serviced_demand: int = self.demand
        self.average_unload_time: float = run_settings.RUN_DATA.average_unload_time[self.data_index]
        # self.expected_unload_time: float = self.serviced_demand * self.average_unload_time

    def __index__(self) -> int:
        return self.data_index

    def __repr__(self):
        return f"Loc {self.name} (Rt {self.route_index}, Ind {self.index_in_route}; Dem {self.demand}, " \
               f"Srv {self.serviced_demand})"
        # f"ID {id(self)} "

    def __eq__(self, other):
        if isinstance(other, Location):
            return other.data_index == self.data_index
            # and other.route_index == self.route_index and other.index_in_route == self.index_in_route
        return False

    # def __copy__(self):
    #     """A copy only needs to care about the serviced demand and demand values, nothing else will change."""
    #     cls = self.__class__
    #     new_object = cls.__new__(cls)
    #     new_object.__dict__.update(self.__dict__)

    def __deepcopy__(self, memo: dict):
        """If included in a deepcopy, do just a shallow copy. There is no need to deepcopy the tuples here."""
        existing = memo.get(id(self))
        if existing:
            return existing
        new_copy = copy(self)
        memo[id(self)] = new_copy
        return new_copy

    @property
    def demand(self) -> int:
        """Another demand value can be provided for locations to be used by decomposed problems, which will then
        override the actual demand for uses such as complete_customer_services."""
        if self._demand:
            return self._demand
        else:
            return self.actual_demand

    @demand.setter
    def demand(self, value: int):
        self._demand = value

    @demand.deleter
    def demand(self):
        self._demand = self.actual_demand

    @property
    def satisfies_all_demand(self) -> bool:
        return self.serviced_demand == self.demand

    @property
    def expected_unload_time(self) -> float:
        return self.serviced_demand * self.average_unload_time

    def find_correlation(self, neighbor: "Location", vehicle_type: "VehicleType") -> float:
        """Finds the correlation between two locations according to equation 4 in section 4.5.1. of Vidal et al. (2012).

        This is to find the relevance of two locations to each other for the neighborhood search.
        Only the most promising locations will be considered in the neighborhood."""
        return (vehicle_type.distance_cost * self.travel_distances[neighbor.data_index]) + \
               (vehicle_type.time_cost *
                (max(0.0, neighbor.window_start - self.expected_unload_time - self.travel_times[neighbor.data_index] -
                     self.window_end) +
                 run_settings.RUN_CONFIG.duration_penalty_multiplier *
                 max(0.0, self.window_start + self.expected_unload_time + self.travel_times[neighbor.data_index] -
                     neighbor.window_end)))


class LocationChecker(Location):
    """A sub-class of ArcLocation, intended to be used when checking that all customers have the required service
    levels """

    def __init__(self, location: Location):
        super().__init__(data_index=location.data_index, serviced_demand=0)
        # This will be a list of tuples with format (vehicle_type, tour_index)
        # self.stop_type_indices: List[Tuple[int, int]] = []
        self.demand = location.demand
        self.excess_capacity = 0


class VehicleType:
    def __init__(self, data_index: int):
        self.data_index = data_index
        self.name: str = run_settings.RUN_DATA.vehicle_types[data_index]
        self.capacity: int = run_settings.RUN_DATA.pallet_capacity[data_index]
        self.distance_cost: float = run_settings.RUN_DATA.distance_cost[data_index]
        self.time_cost: float = run_settings.RUN_DATA.time_cost[data_index]
        self.available_vehicles: int = run_settings.RUN_DATA.available_vehicles[data_index]
        if run_settings.RUN_DATA.hired_cost_multiplier is not None:
            self.hired_cost_multiplier: float = run_settings.RUN_DATA.hired_cost_multiplier[data_index]
        else:
            self.hired_cost_multiplier: float = 1

    def __index__(self) -> int:
        return self.data_index

    def __repr__(self) -> str:
        return f"Vehicle {self.data_index} (\"{self.name}\", Cap {self.capacity})"

    def __eq__(self, other):
        if isinstance(other, VehicleType):
            return self.data_index == other.data_index


class DataGlobals:
    DEPOT: Optional[Location] = None
    # DEPOT_RETURN: Optional[ArcLocation] = None
    ALL_CUSTOMERS: Optional[ndarray] = None
    ALL_VEHICLE_TYPES: Optional[ndarray] = None

    def update_globals(self):
        """Updates the global variables with the current run data."""
        self.DEPOT = Location(0)
        # self.DEPOT_RETURN = ArcLocation(len(run_settings.RUN_DATA.locations) - 1)

        all_customers: List[Location] = [Location(customer_index) for customer_index, _ in
                                         enumerate(run_settings.RUN_DATA.locations)]
        # Only include customers that have demand
        self.ALL_CUSTOMERS: ndarray[Location] = array([customer for customer in all_customers if customer.demand > 0])
        # test = self.ALL_CUSTOMERS.index(209)

        all_vehicle_types: List[VehicleType] = [VehicleType(index) for index, _ in
                                                enumerate(run_settings.RUN_DATA.vehicle_types)]
        # Only include vehicle types that have available vehicles
        self.ALL_VEHICLE_TYPES: ndarray[VehicleType] = array(
            [vehicle_type for vehicle_type in all_vehicle_types if vehicle_type.available_vehicles > 0])


data_globals = DataGlobals()
