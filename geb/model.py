# -*- coding: utf-8 -*-
import datetime
from pathlib import Path
from operator import attrgetter
import geopandas as gpd
from typing import Union
from time import time
import copy
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from honeybees.library.helpers import timeprint
from honeybees.area import Area
from honeybees.model import Model as ABM_Model

from geb.reporter import Reporter
from geb.agents import Agents
from geb.agents.general import AgentArray
from geb.artists import Artists
from geb.HRUs import Data
from geb.cwatm_model import CWatM_Model


class GEBModel(ABM_Model, CWatM_Model):
    """GEB parent class.

    Args:
        config: Filepath of the YAML-configuration file.
        CwatM_settings: Path of CWatM settings file.
        name: Name of model.
        xmin: Minimum x coordinate.
        xmax: Maximum x coordinate.
        ymin: Minimum y coordinate.
        ymax: Maximum y coordinate.
        args: Run arguments.
        coordinate_system: Coordinate system that should be used. Currently only accepts WGS84.
    """

    description = """GEB stands for Geographic Environmental and Behavioural model and is named after Geb, the personification of Earth in Egyptian mythology.\nGEB aims to simulate both environment, for now the hydrological system, the behaviour of people and their interactions at large scale without sacrificing too much detail. The model does so by coupling an agent-based model which simulates millions individual people or households and a hydrological model. While the model can be expanded to other agents and environmental interactions, we focus on farmers, high-level agents, irrigation behaviour and land management for now."""

    def __init__(
        self,
        config: dict,
        model_structure: dict,
        spinup: bool = False,
        use_gpu: bool = False,
        gpu_device=0,
        coordinate_system: str = "WGS84",
    ):
        self.spinup = spinup
        self.use_gpu = use_gpu
        if self.use_gpu:
            cp.cuda.Device(gpu_device).use()

        self.config = self.setup_config(config)
        if self.spinup:
            self.config["report"] = {}

        # make a deep copy to avoid issues when the model is initialized multiple times
        self.model_structure = copy.deepcopy(model_structure)
        for data in self.model_structure.values():
            for key, value in data.items():
                data[key] = Path(config["general"]["input_folder"]) / value

        if spinup is True:
            self.run_name = "spinup"
        elif "name" in self.config["general"]:
            self.run_name = self.config["general"]["name"]
        else:
            print('No "name" specified in config file under general. Using "default".')
            self.run_name = "default"

        self.report_folder = (
            Path(self.config["general"]["report_folder"]) / self.run_name
        )
        self.report_folder.mkdir(parents=True, exist_ok=True)

        self.initial_conditions_folder = Path(
            self.config["general"]["initial_conditions_folder"]
        )
        self.initial_relations_folder = self.initial_conditions_folder / "relations"

        if self.spinup is True:
            end_time = datetime.datetime.combine(
                self.config["general"]["start_time"], datetime.time(0)
            )
            current_time = datetime.datetime.combine(
                self.config["general"]["spinup_time"], datetime.time(0)
            )
            if end_time.year - current_time.year < 10:
                print(
                    "Spinup time is less than 10 years. This is not recommended and may lead to issues later."
                )

            self.load_initial_data = False
            self.save_initial_data = self.config["general"]["export_inital_on_spinup"]
            self.initial_conditions = []
        else:
            current_time = datetime.datetime.combine(
                self.config["general"]["start_time"], datetime.time(0)
            )
            end_time = datetime.datetime.combine(
                self.config["general"]["end_time"], datetime.time(0)
            )
            self.load_initial_data = True
            self.save_initial_data = False

        assert isinstance(end_time, datetime.datetime)
        assert isinstance(current_time, datetime.datetime)

        timestep_length = datetime.timedelta(days=1)
        n_timesteps = (end_time - current_time) / timestep_length
        assert n_timesteps.is_integer()
        n_timesteps = int(n_timesteps)
        assert n_timesteps > 0, "End time is before or identical to start time"

        self.regions = gpd.read_file(self.model_structure["geoms"]["areamaps/regions"])
        self.data = Data(self)

        self.__init_ABM__(
            current_time,
            timestep_length,
            n_timesteps,
            coordinate_system,
        )
        self.__init_hydromodel__(self.config["general"]["CWatM_settings"])
        if self.config["general"]["simulate_floods"]:
            from geb.sfincs import SFINCS

            # exract the longest flood event in days
            flood_events = self.config["general"]["flood_events"]
            flood_event_lengths = [
                event["end_time"] - event["start_time"] for event in flood_events
            ]
            longest_flood_event = max(flood_event_lengths).days
            self.sfincs = SFINCS(
                self, config=self.config, n_timesteps=longest_flood_event
            )

        self.reporter = Reporter(self)

        np.savez_compressed(
            Path(self.reporter.abm_reporter.export_folder, "land_owners.npz"),
            data=self.data.HRU.land_owners,
        )
        np.savez_compressed(
            Path(self.reporter.abm_reporter.export_folder, "unmerged_HRU_indices.npz"),
            data=self.data.HRU.unmerged_HRU_indices,
        )
        np.savez_compressed(
            Path(self.reporter.abm_reporter.export_folder, "scaling.npz"),
            data=self.data.HRU.scaling,
        )
        np.savez_compressed(
            Path(self.reporter.abm_reporter.export_folder, "activation_order.npz"),
            data=self.agents.farmers.activation_order_by_elevation,
        )

    def __init_ABM__(
        self,
        current_time,
        timestep_length,
        n_timesteps,
        coordinate_system: str,
    ) -> None:
        """Initializes the agent-based model.

        Args:
            config_path: Filepath of the YAML-configuration file.
            args: Run arguments.
            coordinate_system: Coordinate system that should be used. Currently only accepts WGS84.
        """

        ABM_Model.__init__(
            self,
            current_time,
            timestep_length,
            args=None,
            n_timesteps=n_timesteps,
        )

        study_area = {
            "xmin": self.data.grid.bounds.left,
            "xmax": self.data.grid.bounds.right,
            "ymin": self.data.grid.bounds.bottom,
            "ymax": self.data.grid.bounds.top,
        }

        self.area = Area(self, study_area)
        self.agents = Agents(self)
        self.artists = Artists(self)

        assert (
            coordinate_system == "WGS84"
        )  # coordinate system must be WGS84. If not, all code needs to be reviewed

        # This variable is required for the batch runner. To stop the model
        # if some condition is met set running to False.
        timeprint("Finished setup")

    def __init_hydromodel__(self, settings: str) -> None:
        """Function to initialize CWatM.

        Args:
            settings: Filepath of CWatM settingsfile
        """
        CWatM_Model.__init__(
            self, self.current_time + self.timestep_length, self.n_timesteps, settings
        )

    def step(self, step_size: Union[int, str] = 1) -> None:
        """
        Forward the model by the given the number of steps.

        Args:
            step_size: Number of steps the model should take. Can be integer or string `day`, `week`, `month`, `year`, `decade` or `century`.
        """

        if isinstance(step_size, str):
            n = self.parse_step_str(step_size)
        else:
            n = step_size
        for _ in range(n):
            print(self.current_time, flush=True)
            t0 = time()
            self.data.step()
            ABM_Model.step(self, 1, report=False)
            CWatM_Model.step(self, 1)

            if self.config["general"]["simulate_floods"]:
                self.sfincs.save_discharge()

                for event in self.config["general"]["flood_events"]:
                    assert type(self.current_time.date()) == type(event["end_time"])
                    if self.current_time.date() == event["end_time"]:
                        self.sfincs.run(event["basin_id"], event["start_time"])

            self.reporter.step()
            t1 = time()
            # print(t1-t0, 'seconds')

    def run(self) -> None:
        """Run the model for the entire period, and export water table in case of spinup scenario."""
        for _ in range(self.n_timesteps):
            self.step()

        if self.save_initial_data:
            self.initial_conditions_folder.mkdir(parents=True, exist_ok=True)
            with open(
                Path(self.initial_conditions_folder, "initial_conditions.txt"), "w"
            ) as f:
                for var in self.initial_conditions:
                    f.write(f"{var}\n")

                    fp = self.initial_conditions_folder / f"{var}.npz"
                    values = attrgetter(var)(self.data)
                    np.savez_compressed(fp, data=values)

            for attribute, value in vars(self.agents.farmers).items():
                if isinstance(value, AgentArray):
                    fp = Path(
                        self.initial_conditions_folder, f"farmers.{attribute}.npz"
                    )
                    np.savez_compressed(fp, data=value.data)

        print("Model run finished")

    @property
    def current_day_of_year(self) -> int:
        """Gets the current day of the year.

        Returns:
            day: current day of the year.
        """
        return self.current_time.timetuple().tm_yday

    @property
    def simulation_root(self) -> Path:
        """Gets the simulation root.

        Returns:
            simulation_root: Path of the simulation root.
        """
        folder = Path("simulation_root")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def close(self) -> None:
        """Finalizes the model."""
        CWatM_Model.finalize(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
