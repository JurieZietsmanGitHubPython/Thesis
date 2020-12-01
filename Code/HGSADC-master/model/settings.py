import math
import os
import statistics
from typing import Tuple, List, Union, Optional

import numpy
from openpyxl import load_workbook
from openpyxl.cell import ReadOnlyCell
from openpyxl.workbook.defined_name import DefinedName

from util import create_logger

logger = create_logger(__name__)


class Data:

    def __init__(self, filename: str = None, locations: List[str] = None, vehicle_types: List[str] = None,
                 distance_cost: List[float] = None, time_cost: List[float] = None, pallet_capacity: List[int] = None,
                 available_vehicles: List[int] = None, hired_cost_multiplier: List[float] = None,
                 demand: List[int] = None, window_start: List[float] = None, window_end: List[float] = None,
                 average_unload_time: List[float] = None, distances: List[List[float]] = None,
                 times: List[List[float]] = None):
        """If filename provided, will read in run_data from .xlsx using OpenPyXL.
        If no filename provided, will check named parameters for values."""
        if filename:
            try:
                self.workbook = load_workbook(filename=filename, read_only=True, data_only=True)

                # one_dimension_sheet: Worksheet = self.workbook["1D"]

                # Indexing arrays
                for title, coord in self.workbook.defined_names["Locations"].destinations:
                    logger.debug(title, coord)
                # logger.debug(self.extract_ranges(self.workbook.defined_names["Locations"])[0])

                self.locations = numpy.array(self.extract_named_range("Locations"))
                # self.stores = self.extract_named_range("Stores")
                self.vehicle_types = numpy.array(self.extract_named_range("VehicleTypes"))
                # self.horses = self.extract_named_range("Horses")
                # self.trailers = self.extract_named_range("Trailers")

                # 1D parameters
                self.distance_cost = numpy.array(self.extract_named_range("DistanceCost"))
                self.time_cost = numpy.array(self.extract_named_range("TimeCost"))
                self.pallet_capacity = numpy.array(self.extract_named_range("PalletCapacity"))
                # print(self.pallet_capacity)
                self.available_vehicles = numpy.array(self.extract_named_range("AvailableVehicles"))
                self.hired_cost_multiplier = numpy.array(self.extract_named_range("HiredCost"))
                self.demand = numpy.array(self.extract_named_range("Demand"))
                self.window_start = numpy.array(self.extract_named_range("WindowStart"))
                self.window_end = numpy.array(self.extract_named_range("WindowEnd"))
                self.average_unload_time = numpy.array(self.extract_named_range("AverageUnload"))

                # 2D parameters
                self.distances = numpy.array(numpy.array(line) for line in self.extract_named_range("Distances"))
                self.times = numpy.array(numpy.array(line) for line in self.extract_named_range("Times"))
                # self.horse_trailer_compatibility: List[List[int]] = self.extract_named_range("HorseTrailer")
                # self.store_horse_compatibility: List[List[int]] = self.extract_named_range("HorseStore")
                # self.store_trailer_compatibility: List[List[int]] = self.extract_named_range("TrailerStore")
            except IOError as err:
                logger.error(err)
        else:
            self.locations: numpy.ndarray[str] = numpy.array(locations)
            self.vehicle_types: numpy.ndarray[str] = numpy.array(vehicle_types)
            self.distance_cost: numpy.ndarray[float] = numpy.array(distance_cost)
            self.time_cost: numpy.ndarray[float] = numpy.array(time_cost)
            self.pallet_capacity: numpy.ndarray[int] = numpy.array(pallet_capacity)
            self.available_vehicles: numpy.ndarray[int] = numpy.array(available_vehicles)
            if hired_cost_multiplier:
                self.hired_cost_multiplier: numpy.ndarray[float] = numpy.array(hired_cost_multiplier)
            else:
                self.hired_cost_multiplier = None
            self.demand: numpy.ndarray[int] = numpy.array(demand)
            self.window_start: numpy.ndarray[float] = numpy.array(window_start)
            self.window_end: numpy.ndarray[float] = numpy.array(window_end)
            self.average_unload_time: numpy.ndarray[float] = numpy.array(average_unload_time)
            self.distances: numpy.ndarray[numpy.ndarray[float]] = numpy.array([numpy.array(sub) for sub in distances])
            self.times: numpy.ndarray[numpy.ndarray[float]] = numpy.array([numpy.array(sub) for sub in times])

    def __repr__(self):
        return f"Data object {id(self)} with {len(self.locations) - 2} customers and {len(self.vehicle_types)} " \
               f"vehicle types."

    def repr_all(self) -> str:
        return f"{vars(self)}"

    def extract_named_range(self, name: str):
        return self.extract_values(self.extract_ranges(self.workbook.defined_names[name])[0])

    def extract_ranges(self, defined_name: DefinedName) -> List[Tuple[Tuple[ReadOnlyCell]]]:
        """Get a list of ranges (composed of tuples of Cell objects) for the defined name range"""
        ranges = []
        for title, coord in defined_name.destinations:
            sheet = self.workbook[title]
            ranges.append(sheet[coord])
        return ranges

    @staticmethod
    def extract_values(cells: Tuple[Tuple[ReadOnlyCell]]) -> Union[list, List[list]]:
        values = []
        for row in cells:
            row_vals = []
            for cell in row:
                row_vals.append(cell.value)
            if len(row_vals) == 1:
                row_vals = row_vals[0]
            values.append(row_vals)
        if len(values) == 1:
            values = values[0]
        return values

    # def dump_to_sheet(self, filename: str, tablename: str):
    #     """Dump model data to excel sheet."""


class Config:

    # def __init__(self, non_improving_iterations: int, max_run_time: timedelta):
    def __init__(self, run_data: Data):
        """Initialize the instance of the algorithm"""
        # self.non_improving_iterations = non_improving_iterations
        # self.max_run_time = max_run_time
        # self.diversification_proportion = diversification_proportion
        # self.min_pop_size = min_pop_size
        # self.generation_size = generation_size
        # self.decomposition_iterations = decomposition_iterations

        # Calibration parameters
        self.min_pop_size = 25
        self.generation_size = 100
        self.proportion_elite = 0.4
        self.proportion_close_eval = 0.2

        self.education_rate = 1.0
        self.repair_rate = 0.5
        self.granularity_threshold = 0.4
        self.neighborhood_improvement_gap = 0.05
        self.procedure_time_limit = 1  # Seconds

        self.reference_feasible_proportion = 0.2
        self.diversification_proportion = 0.4

        self.min_customers_for_decomposition = 60
        self.decomposition_iterations = 1000

        self.n_close = math.floor(self.min_pop_size * self.proportion_close_eval)

        # Penalty parameter multipliers
        self.parameter_adjustment_frequency = 100
        self.duration_penalty_multiplier = 1
        if run_data:
            distance_row_means = [statistics.mean(row) for row in run_data.distances]
            self.capacity_penalty_multiplier = statistics.mean(distance_row_means) / statistics.mean(
                run_data.demand[1:])

        # Other parameters
        # Should the algorithm consider rest time in its cost evaluations
        self.consider_rest_time = True
        # The maximum number of hours in a single day
        self.daily_drive_limit = 10
        # The number of hours to rest between days
        self.daily_rest = 8
        # The maximum number of hours driving without rest
        self.continuous_drive_limit = 5
        # The amount of time to rest after continuously driving for too long
        self.continuous_rest = 0.45
        # The largest gap between the start and end time of a store's "service window" for it to count as a window
        # when finding the departure time (during the algorithm, the departure time will be set so that the
        # vehicle_index arrives as the window opens for the first store with a window)
        self.window_threshold = 2
        # If a positive number, solutions which do not return to the depot by this time will be penalised with warp time
        self.return_to_depot_before = -1
        # The number of CPU cores this process has access to
        # self.available_cores = len(os.sched_getaffinity(0))
        # The number of CPU cores the computer has
        self.available_cores = os.cpu_count()
        # What proportion of the CPU's cores should be used if using multithreading
        self.core_proportion = 1
        # Allow the algorithm to use as many vehicles as it wants
        self.allow_unlimited_fleets = True
        # The number of hours that vehicles that have returned from routes will "wait" before beginning another route
        # This is added to account for time loss during/after routes, along with the fact that vehicles won't
        # immediately begin loading for then next route as soon as they arrive at the depot.
        self.waste_between_routes = 1
        # The hours of service to expect from a single vehicle per day
        self.daily_service_time = 16
        # The range in km in which stores are considered "local".
        # Customers beyond the local range will only be serviced by third party vehicles.
        self.local_range = 150

    def __repr__(self):
        return f"Config object {id(self)}"

    def adjust_penalties(self, capacity_feasible_proportion: float, duration_feasible_proportion: float):
        """Readjusts the capacity or duration penalty parameters according to the proportions of infeasible solutions
        with respect to the target proportion of naturally-feasible solutions.

        This should be run every 100 solutions, and the naturally feasible proportions should be based on these 100
        solutions."""
        if capacity_feasible_proportion <= self.reference_feasible_proportion - 0.05:
            self.capacity_penalty_multiplier *= 1.2
        elif capacity_feasible_proportion >= self.reference_feasible_proportion + 0.05:
            self.capacity_penalty_multiplier *= 0.85

        if duration_feasible_proportion <= self.reference_feasible_proportion - 0.05:
            self.duration_penalty_multiplier *= 1.2
        elif duration_feasible_proportion >= self.reference_feasible_proportion + 0.05:
            self.duration_penalty_multiplier *= 0.85


class Settings:
    RUN_DATA: Optional[Data] = None
    RUN_CONFIG: Optional[Config] = None

    # run_data = Data("Model Data.xlsx")
    # run_config = Config()

    def set_run_data(self, to_set: Data):
        logger.debug(f"Setting run data {to_set}")

        if to_set:
            self.RUN_DATA = to_set
            self.RUN_CONFIG = Config(to_set)

            from data_objects import data_globals
            data_globals.update_globals()
