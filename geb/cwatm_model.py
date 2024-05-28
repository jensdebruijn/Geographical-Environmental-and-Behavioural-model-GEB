# -*- coding: utf-8 -*-
import numpy as np
import os
import datetime
from pathlib import Path

from cwatm.cwatm_model import CWATModel
from cwatm.management_modules.dynamicModel import ModelFrame
from cwatm.management_modules.configuration import parse_configuration
from cwatm.management_modules.globals import settingsfile, binding, outDir
from cwatm.run_cwatm import headerinfo


class CWatM_Model(CWATModel):
    """
    This class is used to initalize the CWatM model from GEB. Several static configuration files are read first, then several dynamic parameters are set based on the configuration of GEB. Then, the model frame is created that can then later be used to iteratate.

    Args:
        start_time: Start date of the model.
        n_steps: Number of steps that the model will run for.
        settings: Filepath of the CWatM settingsfile. For full configuration options please refer to the `CWatM documentation <https://cwatm.iiasa.ac.at/>`.
        use_gpu: Whether the model can use a GPU.
    """

    def __init__(
        self, start_time: datetime.datetime, n_steps: int, settings: str
    ) -> None:
        self.init_water_table_file = os.path.join(
            self.config["general"]["init_water_table"]
        )

        settingsfile.append(settings)
        parse_configuration(settings)

        outDir["OUTPUT"] = self.report_folder

        # calibration
        for parameter, value in self.config["parameters"].items():
            binding[parameter] = value

        binding["MaskMap"] = self.model_structure["grid"]["areamaps/grid_mask"]
        if "gauges" in self.config["general"]:
            gauges = self.config["general"]["gauges"]
            binding["Gauges"] = " ".join(
                [str(item) for sublist in gauges for item in sublist]
            )
        else:
            binding["Gauges"] = (
                f"{self.config['general']['pour_point'][0]} {self.config['general']['pour_point'][1]}"
            )
        binding["StepStart"] = start_time.strftime("%d/%m/%Y")
        binding["SpinUp"] = "0"
        binding["StepEnd"] = str(n_steps)
        binding["Modflow_resolution"] = 1000

        # setting file paths for CWatM
        binding["Ldd"] = self.model_structure["grid"]["routing/kinematic/ldd"]
        binding["ElevationStD"] = self.model_structure["grid"][
            "landsurface/topo/elevation_STD"
        ]
        binding["elevation"] = self.model_structure["grid"][
            "landsurface/topo/elevation"
        ]
        binding["CellArea"] = self.model_structure["grid"]["areamaps/cell_area"]
        binding["cropgroupnumber"] = self.model_structure["grid"]["soil/cropgrp"]
        binding["KSat1"] = self.model_structure["grid"]["soil/ksat1"]
        binding["KSat2"] = self.model_structure["grid"]["soil/ksat2"]
        binding["KSat3"] = self.model_structure["grid"]["soil/ksat3"]
        binding["alpha1"] = self.model_structure["grid"]["soil/alpha1"]
        binding["alpha2"] = self.model_structure["grid"]["soil/alpha2"]
        binding["alpha3"] = self.model_structure["grid"]["soil/alpha3"]
        binding["lambda1"] = self.model_structure["grid"]["soil/lambda1"]
        binding["lambda2"] = self.model_structure["grid"]["soil/lambda2"]
        binding["lambda3"] = self.model_structure["grid"]["soil/lambda3"]
        binding["thetas1"] = self.model_structure["grid"]["soil/thetas1"]
        binding["thetas2"] = self.model_structure["grid"]["soil/thetas2"]
        binding["thetas3"] = self.model_structure["grid"]["soil/thetas3"]
        binding["thetar1"] = self.model_structure["grid"]["soil/thetar1"]
        binding["thetar2"] = self.model_structure["grid"]["soil/thetar2"]
        binding["thetar3"] = self.model_structure["grid"]["soil/thetar3"]
        binding["forest_fao_ksat1"] = self.model_structure["grid"]["soil/forest_fao_ksat1"]
        binding["forest_fao_ksat2"] = self.model_structure["grid"]["soil/forest_fao_ksat2"]
        binding["forest_fao_ksat3"] = self.model_structure["grid"]["soil/forest_fao_ksat3"]
        binding["forest_fao_alpha1"] = self.model_structure["grid"]["soil/forest_fao_alpha1"]
        binding["forest_fao_alpha2"] = self.model_structure["grid"]["soil/forest_fao_alpha2"]
        binding["forest_fao_alpha3"] = self.model_structure["grid"]["soil/forest_fao_alpha3"]
        binding["forest_fao_lambda1"] = self.model_structure["grid"]["soil/forest_fao_lambda1"]
        binding["forest_fao_lambda2"] = self.model_structure["grid"]["soil/forest_fao_lambda2"]
        binding["forest_fao_lambda3"] = self.model_structure["grid"]["soil/forest_fao_lambda3"]
        binding["forest_fao_thetas1"] = self.model_structure["grid"]["soil/forest_fao_thetas1"]
        binding["forest_fao_thetas2"] = self.model_structure["grid"]["soil/forest_fao_thetas2"]
        binding["forest_fao_thetas3"] = self.model_structure["grid"]["soil/forest_fao_thetas3"]
        binding["forest_fao_thetar1"] = self.model_structure["grid"]["soil/forest_fao_thetar1"]
        binding["forest_fao_thetar2"] = self.model_structure["grid"]["soil/forest_fao_thetar2"]
        binding["forest_fao_thetar3"] = self.model_structure["grid"]["soil/forest_fao_thetar3"]
        binding["agr_fao_alpha1"] = self.model_structure["grid"]["soil/agr_fao_alpha1"]
        binding["agr_fao_alpha2"] = self.model_structure["grid"]["soil/agr_fao_alpha2"]
        binding["agr_fao_alpha3"] = self.model_structure["grid"]["soil/agr_fao_alpha3"]
        binding["agr_fao_KSat1"] = self.model_structure["grid"]["soil/agr_fao_ksat1"]
        binding["agr_fao_KSat2"] = self.model_structure["grid"]["soil/agr_fao_ksat2"]
        binding["agr_fao_KSat3"] = self.model_structure["grid"]["soil/agr_fao_ksat3"]
        binding["agr_fao_lambda1"] = self.model_structure["grid"]["soil/agr_fao_lambda1"]
        binding["agr_fao_lambda2"] = self.model_structure["grid"]["soil/agr_fao_lambda2"]
        binding["agr_fao_lambda3"] = self.model_structure["grid"]["soil/agr_fao_lambda3"]
        binding["agr_fao_thetar1"] = self.model_structure["grid"]["soil/agr_fao_thetar1"]
        binding["agr_fao_thetar2"] = self.model_structure["grid"]["soil/agr_fao_thetar2"]
        binding["agr_fao_thetar3"] = self.model_structure["grid"]["soil/agr_fao_thetar3"]
        binding["agr_fao_thetas1"] = self.model_structure["grid"]["soil/agr_fao_thetas1"]
        binding["agr_fao_thetas2"] = self.model_structure["grid"]["soil/agr_fao_thetas2"]
        binding["agr_fao_thetas3"] = self.model_structure["grid"]["soil/agr_fao_thetas3"]
        binding["grs_fao_ksat2"] = self.model_structure["grid"]["soil/grs_fao_ksat2"]  
        binding["slope"] = self.model_structure["grid"]["landsurface/topo/slope"]

        binding["forest_types"] = self.model_structure["grid"]["landsurface/forest_types"]

        binding["percolationImp"] = self.model_structure["grid"][
            "soil/percolation_impeded"
        ]
        binding["StorDepth1"] = self.model_structure["grid"]["soil/storage_depth1"]
        binding["StorDepth2"] = self.model_structure["grid"]["soil/storage_depth2"]

        binding["forest_KSat1"] = self.model_structure["grid"]["soil/ksat1"]
        binding["forest_KSat2"] = self.model_structure["grid"]["soil/ksat2"]
        binding["forest_KSat3"] = self.model_structure["grid"]["soil/ksat3"]
        binding["forest_alpha1"] = self.model_structure["grid"]["soil/alpha1"]
        binding["forest_alpha2"] = self.model_structure["grid"]["soil/alpha2"]
        binding["forest_alpha3"] = self.model_structure["grid"]["soil/alpha3"]
        binding["forest_lambda1"] = self.model_structure["grid"]["soil/lambda1"]
        binding["forest_lambda2"] = self.model_structure["grid"]["soil/lambda2"]
        binding["forest_lambda3"] = self.model_structure["grid"]["soil/lambda3"]
        binding["forest_thetas1"] = self.model_structure["grid"]["soil/thetas1"]
        binding["forest_thetas2"] = self.model_structure["grid"]["soil/thetas2"]
        binding["forest_thetas3"] = self.model_structure["grid"]["soil/thetas3"]
        binding["forest_thetar1"] = self.model_structure["grid"]["soil/thetar1"]
        binding["forest_thetar2"] = self.model_structure["grid"]["soil/thetar2"]
        binding["forest_thetar3"] = self.model_structure["grid"]["soil/thetar3"]

        binding["forest_rootFraction1"] = self.model_structure["grid"][
            "landcover/forest/rootFraction1_forest"
        ]
        binding["forest_maxRootDepth"] = self.model_structure["grid"][
            "landcover/forest/maxRootDepth_forest"
        ]

        binding["grassland_rootFraction1"] = self.model_structure["grid"][
            "landcover/grassland/rootFraction1_grassland"
        ]
        binding["grassland_maxRootDepth"] = self.model_structure["grid"][
            "landcover/grassland/maxRootDepth_grassland"
        ]
        binding["grassland_cropCoefficientNC"] = self.model_structure["forcing"][
            "landcover/grassland/cropCoefficientGrassland_10days"
        ]

        binding["irrPaddy_rootFraction1"] = self.model_structure["grid"][
            "landcover/irrPaddy/rootFraction1_irrPaddy"
        ]
        binding["irrPaddy_maxRootDepth"] = self.model_structure["grid"][
            "landcover/irrPaddy/maxRootDepth_irrPaddy"
        ]
        binding["irrPaddy_cropCoefficientNC"] = self.model_structure["forcing"][
            "landcover/irrPaddy/cropCoefficientirrPaddy_10days"
        ]

        binding["irrNonPaddy_rootFraction1"] = self.model_structure["grid"][
            "landcover/irrNonPaddy/rootFraction1_irrNonPaddy"
        ]
        binding["irrNonPaddy_maxRootDepth"] = self.model_structure["grid"][
            "landcover/irrNonPaddy/maxRootDepth_irrNonPaddy"
        ]
        binding["irrNonPaddy_cropCoefficientNC"] = self.model_structure["forcing"][
            "landcover/irrNonPaddy/cropCoefficientirrNonPaddy_10days"
        ]

        binding["chanMan"] = self.model_structure["grid"]["routing/kinematic/mannings"]
        binding["chanLength"] = self.model_structure["grid"][
            "routing/kinematic/channel_length"
        ]
        binding["chanWidth"] = self.model_structure["grid"][
            "routing/kinematic/channel_width"
        ]
        binding["chanDepth"] = self.model_structure["grid"][
            "routing/kinematic/channel_depth"
        ]
        binding["chanRatio"] = self.model_structure["grid"][
            "routing/kinematic/channel_ratio"
        ]
        binding["chanGrad"] = self.model_structure["grid"][
            "routing/kinematic/channel_slope"
        ]

        binding["waterBodyID"] = self.model_structure["grid"][
            "routing/lakesreservoirs/lakesResID"
        ]
        binding["reservoir_command_areas"] = self.model_structure["grid"][
            "routing/lakesreservoirs/command_areas"
        ]

        binding["PathGroundwaterModflow"] = "modflow"

        headerinfo()

        CWATModel.__init__(self)
        self.stCWATM = ModelFrame(self)

    def step(self, n: int) -> None:
        """Performs n number of (daily) steps in CWatM.

        Args:
            n: Number of timesteps to perform.
        """
        for _ in range(n):
            self.stCWATM.step()

    def finalize(self) -> None:
        """Finalizes CWatM."""
        self.stCWATM.finalize()

    def export_water_table(self) -> None:
        """Function to save required water table output to file."""
        dirname = os.path.dirname(self.init_water_table_file)
        os.makedirs(dirname, exist_ok=True)
        np.save(
            self.init_water_table_file,
            self.groundwater_modflow_module.modflow.decompress(
                self.groundwater_modflow_module.modflow.head
            ),
        )
