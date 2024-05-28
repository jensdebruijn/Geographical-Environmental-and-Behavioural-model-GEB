# -*- coding: utf-8 -*-
from typing import Union
from honeybees.artists import Artists as honeybeesArtists
import numpy as np
import re
from operator import attrgetter

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass

from geb.agents.farmers import FarmerAgentArray


class Artists(honeybeesArtists):
    """This class is used to configure how the display environment works.

    Args:
        model: The GEB model.
    """

    def __init__(self, model) -> None:
        honeybeesArtists.__init__(self, model)
        self.color = "#1386FF"
        self.min_colorbar_alpha = 0.4
        self.background_variable = (
            "data.HRU.land_use_type"  # set initial background iamge.
        )
        self.custom_plot = self.get_custom_plot()
        self.set_variables()

    def draw_farmers(self, model, agents, idx: int, color: str = "#ff0000") -> dict:
        """This function is used to draw farmers. First it is determined what crop is grown by the farmer, then the we get the color used to display that crop from the model configuration.

        Args:
            model: The GEB model.
            agents: The farmer class to plot.
            idx: The farmer index.

        Returns:
            portrayal: Portrayal of farmer.
        """
        n_crops = (model.agents.farmers.crops[idx] != -1).sum()
        if n_crops == 1:
            color = "#ff0000"
        elif n_crops == 2:
            color = "#00ff00"
        else:
            color = "#0000ff"
        # if self.model.agents.farmers.flooded[idx] == True:
        #     color = '#ff0000'
        # else:
        #     color = '#0000ff'
        # if idx == self.model.agents.farmers.sample[0]:
        #     color = '#ff0000'
        #     r = 3
        # elif idx == self.model.agents.farmers.sample[1]:
        #     color = '#00ff00'
        #     r = 3
        # elif idx == self.model.agents.farmers.sample[2]:
        #     color = '#0000ff'
        #     r = 3
        # else:
        r = 0.5
        return {
            "type": "shape",
            "shape": "circle",
            "r": r,
            "filled": True,
            "color": color,
        }

    def draw_tehsil(self, properties):
        return {
            "type": "shape",
            "shape": "polygon",
            "filled": False,
            "color": properties["color"],
            "edge": True,
            "linewidth": 2,
        }

    def get_custom_plot(self) -> dict[dict]:
        """Here you can specify custom options for plotting the background.

        Returns:
            custom_dict: Dictionary of dictionaries. The first level is the name of each of the variables, the second level the options for those variables.

        Example:
            .. code-block:: python

                {
                    'HRU.crop_map': {
                        'type': 'discrete',
                        'nanvalue': -1,
                        'names': ['crop name 1', 'crop name 2'],
                        'colors': ['#00FF00', '#FF0000']
                    }
                }

        """
        return {
            "data.HRU.crop_stage": {"type": "discrete"},
            "data.HRU.crop_age": {"type": "discrete"},
            "data.HRU.crop_map": {
                "type": "categorical",
                "nanvalue": -1,
                "names": [
                    self.model.config["draw"]["crop_data"][i]["name"] for i in range(13)
                ],
                "colors": [
                    self.model.config["draw"]["crop_data"][i]["color"]
                    for i in range(13)
                ],
            },
            "data.HRU.land_use_type": {
                "type": "categorical",
                "nanvalue": -1,
                "names": [
                    "forest",
                    "grassland/non-irrigated",
                    "paddy-irrigated",
                    "non-paddy irrigated",
                    "sealed",
                    "water",
                ],
                "colors": [
                    "#274e2e",
                    "#adffbc",
                    "#8555aa",
                    "#007d13",
                    "#7e8180",
                    "#2636d9",
                ],
            },
        }

    def set_variables(self) -> None:
        """This function is used to get a dictionary of variables that can be shown as background variable. The dictionary :code:`self.variables_dict` contains the name of each variable to display as key, and the actual variable as value.

        Checks are performed to see whether the data is the right size. Only compressed data can be shown. If a dataset has multiple dimensions, the dimensions can be shown seperately as `variable[0], variable[1], ...`.
        """
        self.variables_dict = {}

        def add_vars(name, compressed_size, dtypes, variant_dim, invariant_dim):
            assert np.intersect1d(variant_dim, invariant_dim).size == 0
            container = attrgetter(name)(self.model)
            for varname, variable in vars(container).items():
                if isinstance(variable, dtypes):
                    if variable.ndim == 1 and variable.size == compressed_size:
                        self.variables_dict[f"{name}.{varname}"] = variable
                    if (
                        variable.ndim == 2
                        and variable.shape[invariant_dim] == compressed_size
                    ):
                        for i in range(variable.shape[variant_dim]):
                            self.variables_dict[f"{name}.{varname}[{i}]"] = variable[i]
                    else:
                        continue

        add_vars(
            "data.grid",
            compressed_size=self.model.data.grid.compressed_size,
            dtypes=(np.ndarray, cp.ndarray) if self.model.use_gpu else np.ndarray,
            variant_dim=0,
            invariant_dim=1,
        )
        add_vars(
            "data.HRU",
            compressed_size=self.model.data.HRU.compressed_size,
            dtypes=(np.ndarray, cp.ndarray) if self.model.use_gpu else np.ndarray,
            variant_dim=0,
            invariant_dim=1,
        )
        add_vars(
            "agents.farmers",
            compressed_size=self.model.agents.farmers.n,
            dtypes=FarmerAgentArray,
            variant_dim=1,
            invariant_dim=0,
        )

    def get_background_variables(self) -> list:
        """This function gets a list of variables that can be used to show in the background.

        Returns:
            options: List of names for options to show in background.
        """
        self.set_variables()
        return list(self.variables_dict.keys())

    def set_background_variable(self, option_name: str) -> None:
        """This function is used to update the name of the variable to use for drawing the background of the map."""
        self.background_variable = option_name

    def get_background(
        self,
        minvalue: Union[float, int, None] = None,
        maxvalue: Union[float, int, None] = None,
        color: str = "#1386FF",
    ) -> tuple[np.ndarray, dict]:
        """This function is called from the canvas class to draw the canvas background. The name of the variable to draw is stored in `self.background_variable`.

        Args:
            minvalue: The minimum value for the display scale.
            maxvalue: The maximum value for the display scale.
            color: The color to use to display the variable.

        Returns:
            background: RGBA-array to display as background.
            legend: Dictionary with data and formatting rules for background legend.
        """
        if self.background_variable.startswith("agents.farmers"):
            slicer = re.search(r"\[([0-9]+)\]$", self.background_variable)
            if slicer:
                array = attrgetter(self.background_variable[: slicer.span(0)[0]])(
                    self.model
                )
                array = FarmerAgentArray(
                    array[:, int(slicer.group(1))], n=array.shape[0]
                )
            else:
                array = attrgetter(self.background_variable)(self.model)

            mask = self.model.data.HRU.mask
        else:
            compressed_array, array = self.model.reporter.cwatmreporter.get_array(
                self.background_variable, decompress=True
            )
            mask = attrgetter(".".join(self.background_variable.split(".")[:-1]))(
                self.model
            ).mask

        if self.background_variable in self.custom_plot:
            options = self.custom_plot[self.background_variable]
        else:
            options = {}
        if "type" not in options:
            if np.issubsctype(array, np.floating):
                options["type"] = "continuous"
                options["nanvalue"] = np.nan
            elif np.issubsctype(array, np.integer):
                if np.unique(array).size < 30:
                    options["type"] = "categorical"
                    options["nanvalue"] = -1
                else:
                    print(
                        "Type for array might be categorical, but more than 30 categories were found, so rendering as continous."
                    )
                    options["type"] = "continuous"
                    array = array.astype(np.float64)
            elif np.issubsctype(array, bool):
                options["type"] = "bool"
                options["nanvalue"] = -1
            else:
                raise ValueError

        if self.background_variable.startswith("agents.farmers"):
            compressed_array = array.copy()
            array = array.by_field(self.model.data.HRU.land_owners, options["nanvalue"])
            array = self.model.data.HRU.decompress(array)

        if options["type"] == "bool":
            minvalue, maxvalue = 0, 1
        else:
            if not maxvalue:
                maxvalue = np.nanmax(array[~mask], initial=0).item()
            if not minvalue:
                minvalue = np.nanmin(array[~mask], initial=0).item()
            if np.isnan(maxvalue):  # minvalue must be nan as well
                minvalue, maxvalue = 0, 0

        background = np.zeros((*array.shape, 4), dtype=np.uint8)
        if options["type"] == "continuous":
            array -= minvalue
            if maxvalue - minvalue != 0:
                array *= 255 / (maxvalue - minvalue)
            else:
                array *= 0
            array[array < 0] = 0
            array[array > 255] = 255
            rgb = self.hex_to_rgb(color)
            for channel in (0, 1, 2):
                background[:, :, channel][~np.isnan(array)] = rgb[channel] * 255
            background[:, :, 3] = array
            background[:, :, 0][np.isnan(array)] = 200
            background[:, :, 1][np.isnan(array)] = 200
            background[:, :, 2][np.isnan(array)] = 200
            background[:, :, 3][np.isnan(array)] = 255
            legend = {
                "type": "colorbar",
                "color": color,
                "min": self.round_to_n_significant_digits(minvalue, 3),
                "max": self.round_to_n_significant_digits(maxvalue, 3),
                "min_colorbar_alpha": 0,
                "unit": "",
            }
        else:
            if "nanvalue" in options:
                nanvalue = options["nanvalue"]
            else:
                nanvalue = None
            if options["type"] == "categorical":
                unique_values = np.unique(compressed_array)
                if nanvalue is not None:
                    unique_values = unique_values[unique_values != nanvalue]
                unique_values = unique_values.tolist()
                if unique_values:  # no data to be shown on map
                    if "colors" in options:
                        colors = np.array(options["colors"])[
                            np.array(unique_values)
                        ].tolist()
                        colors = [self.hex_to_rgb(color) for color in colors]
                    else:
                        colors = self.generate_distinct_colors(
                            len(unique_values), mode="rgb"
                        )
                    if "names" in options:
                        names = np.array(options["names"])[
                            np.array(unique_values)
                        ].tolist()
                    else:
                        names = unique_values
                else:
                    colors = []
                    names = []
                channels = (0, 1, 2)
                background[:, :, 3][array != nanvalue] = 255
            elif options["type"] == "discrete":
                unique_values = np.arange(
                    compressed_array[compressed_array != nanvalue].min(),
                    compressed_array[compressed_array != nanvalue].max() + 1,
                    1,
                ).tolist()
                if "names" in options:
                    names = options["names"]
                else:
                    names = unique_values
                colors = self.generate_discrete_colors(
                    len(unique_values),
                    self.hex_to_rgb(color),
                    mode="rgb",
                    min_alpha=0.4,
                )
                channels = (0, 1, 2, 3)
            elif options["type"] == "bool":
                unique_values = [False, True]
                names = ["False", "True"]
                channels = (0, 1, 2, 3)
                colors = [(1, 0, 0, 1), (0, 1, 0, 1)]
            else:
                raise ValueError

            if unique_values:
                assert np.all(np.diff(unique_values) > 0)  # check if array is sorted
                for channel in channels:
                    channel_colors = np.array(
                        [color[channel] * 255 for color in colors]
                    )
                    color_array_size = unique_values[-1] + 1
                    if unique_values[0] < 0:
                        color_array_size += abs(unique_values[0])
                    color_array = np.zeros(color_array_size, dtype=np.float32)
                    color_array[np.array(unique_values).astype(np.int32)] = (
                        channel_colors
                    )
                    background[:, :, channel] = color_array[array.astype(np.int32)]

            legend = {
                "type": "legend",
                "labels": {
                    name: self.rgb_to_hex(colors[i]) for i, name in enumerate(names)
                },
            }

        background[mask] = 200

        return background, legend
