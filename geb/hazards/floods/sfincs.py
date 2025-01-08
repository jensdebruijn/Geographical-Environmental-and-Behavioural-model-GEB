from pathlib import Path
from collections import deque
from datetime import datetime
import xarray as xr
import zarr
import numpy as np
import pandas as pd
import geopandas as gpd
from hydromt_sfincs import SfincsModel
import hydromt
from shapely.geometry import Point
from rasterio.features import shapes
import math
import matplotlib.pyplot as plt


try:
    from geb_hydrodynamics.build_model import build_sfincs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'GEB-hydrodynamics' package is not installed. Please install it by running 'pip install geb-hydrodynamics'."
    )
from geb_hydrodynamics.sfincs_utils import run_sfincs_simulation
from geb_hydrodynamics.update_model_forcing import update_sfincs_model_forcing
from geb_hydrodynamics.run_sfincs_for_return_periods import (
    run_sfincs_for_return_periods,
)
from geb_hydrodynamics.postprocess_model import read_flood_map
from geb_hydrodynamics.estimate_discharge_for_return_periods import (
    estimate_discharge_for_return_periods,
)


class SFINCS:
    def __init__(self, model, config, n_timesteps=10):
        self.model = model
        self.config = config
        self.n_timesteps = n_timesteps

        self.discharge_per_timestep = deque(maxlen=self.n_timesteps)

    def sfincs_model_root(self, basin_id):
        folder = self.model.simulation_root / "SFINCS" / str(basin_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def sfincs_simulation_root(self, basin_id):
        folder = (
            self.sfincs_model_root(basin_id)
            / "simulations"
            / f"{self.model.current_time.strftime('%Y%m%dT%H%M%S')}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_event_name(self, event):
        if "basin_id" in event:
            return event["basin_id"]
        elif "region" in event:
            return "region"
        else:
            raise ValueError(
                "Either 'basin_id' or 'region' must be specified in the event."
            )

    @property
    def data_catalogs(self):
        return [
            str(
                Path(self.model.config["general"]["input_folder"])
                / "hydrodynamics"
                / "data_catalog.yml"
            )
        ]

    def get_utm_zone(self, region_file):
        region = gpd.read_file(region_file)
        # Step 1: Calculate the central longitude of the dataset
        centroid = region.geometry.centroid
        central_lon = centroid.x.mean()  # Mean longitude of the dataset

        # Step 2: Determine the UTM zone based on the longitude
        utm_zone = int((central_lon + 180) // 6) + 1

        # Step 3: Determine if the data is in the Northern or Southern Hemisphere
        # The EPSG code for UTM in the northern hemisphere is EPSG:326xx (xx = zone)
        # The EPSG code for UTM in the southern hemisphere is EPSG:327xx (xx = zone)
        if centroid.y.mean() > 0:
            utm_crs = f"EPSG:326{utm_zone}"  # Northern hemisphere
        else:
            utm_crs = f"EPSG:327{utm_zone}"  # Southern hemisphere
        return utm_crs

    def get_detailed_catchment_outline(self, region_file):
        region = gpd.read_file(region_file)
        utm_zone = self.get_utm_zone(region_file)
        region = region.to_crs(utm_zone)
        area = region.geometry.area.item()
        area = area / 1000000
        rounded_area = math.floor(area / 100) * 100  # Round down to nearest hundred

        def vectorize(data, nodata, transform, crs, name="value"):
            feats_gen = shapes(data, mask=data != nodata, transform=transform)
            feats = [
                {"geometry": geom, "properties": {name: val}} for geom, val in feats_gen
            ]
            gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
            gdf[name] = gdf[name].astype(data.dtype)
            return gdf

        # Initialize datacatalog with all sfincs data
        sf = SfincsModel(data_libs=self.data_catalogs)
        merit_hydro = sf.data_catalog.get_rasterdataset(
            "merit_hydro", variables=["flwdir"], geom=region, buffer=100
        )  # get flow directions from merit hydro dataset
        flw = hydromt.flw.flwdir_from_da(
            merit_hydro, ftype="d8", check_ftype=True, mask=None
        )  # Put it in the correct data type
        min_area = rounded_area  # TODO:check whether this would work for other catchments as well
        subbas, idxs_out = flw.subbasins_area(
            min_area
        )  # Extract basin based on minimum upstream area
        catchment_outline = vectorize(
            subbas.astype(np.int32),
            nodata=0,
            transform=flw.transform,
            crs=4326,
            name="basin",
        )  # Vectorize the basin
        catchment_outline = catchment_outline.to_crs(
            utm_zone
        )  # TODO: check if this is the correct crs

        def extract_middle_basin(gdf):
            # Calculate the centroid of each basin
            gdf["centroid"] = gdf.geometry.centroid

            # Calculate the geometric center (mean of centroids) of all basins
            mean_x = gdf.centroid.x.mean()
            mean_y = gdf.centroid.y.mean()
            geometric_center = Point(mean_x, mean_y)

            # Find the basin whose centroid is closest to the geometric center
            gdf["distance_to_center"] = gdf.centroid.apply(
                lambda x: x.distance(geometric_center)
            )
            middle_basin = gdf.loc[gdf["distance_to_center"].idxmin()]

            # Return the middle basin as a new GeoDataFrame
            middle_basin_gdf = gpd.GeoDataFrame([middle_basin], geometry="geometry")

            # Remove the temporary columns before returning
            middle_basin_gdf = middle_basin_gdf.drop(
                columns=["centroid", "distance_to_center"]
            )
            middle_basin_gdf = middle_basin_gdf.set_crs(utm_zone)
            return middle_basin_gdf

        detailed_region = extract_middle_basin(
            catchment_outline
        )  # We're only interested in the basin which is in the middle
        detailed_region.to_file("detailed_region.gpkg", driver="GPKG")
        detailed_region = detailed_region.to_crs(4326)
        return detailed_region

    def setup(self, event, config_fn="sfincs.yml", force_overwrite=False):
        build_parameters = {}

        if "basin_id" in event:
            event_name = self.get_event_name(event)
        elif "region" in event:
            if event["region"] is None:
                model_bbox = self.model.data.grid.bounds
                build_parameters["bbox"] = (
                    model_bbox[0] + 0.1,
                    model_bbox[1] + 0.1,
                    model_bbox[2] - 0.1,
                    model_bbox[3] - 0.1,
                )
            else:
                raise NotImplementedError
            event_name = self.get_event_name(event)
        else:
            raise ValueError(
                "Either 'basin_id' or 'region' must be specified in the event."
            )

        if (
            "simulate_coastal_floods" in self.model.config["general"]
            and self.model.config["general"]["simulate_coastal_floods"]
        ):
            build_parameters["simulate_coastal_floods"] = True

        detailed_region = self.get_detailed_catchment_outline(
            region_file=self.model.files["geoms"]["region"]
        )

        build_parameters.update(
            {
                "config_fn": str(config_fn),
                "model_root": self.sfincs_model_root(event_name),
                "data_catalogs": self.data_catalogs,
                "region": detailed_region,
                "river_method": "default",
                "depth_calculation": "power_law",
                "discharge_ds": self.discharge_spinup_ds,
                "discharge_name": "discharge_daily",
                "uparea_ds": self.uparea_ds,
                "uparea_name": "data",
            }
        )
        if (
            force_overwrite
            or not (self.sfincs_model_root(event_name) / "sfincs.inp").exists()
        ):
            build_sfincs(**build_parameters)

    def to_sfincs_datetime(self, dt: datetime):
        return dt.strftime("%Y%m%d %H%M%S")

    def forecasts(self): # this was a test

    def set_forcing(self, event, start_time):
        if self.model.config["general"]["simulate_hydrology"]:
            n_timesteps = min(self.n_timesteps, len(self.discharge_per_timestep))
            substeps = self.discharge_per_timestep[0].shape[0]
            discharge_grid = self.model.data.grid.decompress(
                np.vstack(self.discharge_per_timestep)
            )

            # when SFINCS starts with high values, this leads to numerical instabilities. Therefore, we first start with very low discharge and then build up slowly to timestep 0
            # TODO: Check if this is a right approach
            discharge_grid = np.vstack(
                [
                    np.full_like(discharge_grid[:substeps, :, :], fill_value=np.nan),
                    discharge_grid,
                ]
            )  # prepend zeros
            for i in range(substeps - 1, -1, -1):
                discharge_grid[i] = discharge_grid[i + 1] * 0.3
            # convert the discharge grid to an xarray DataArray
            discharge_grid = xr.DataArray(
                data=discharge_grid,
                coords={
                    "time": pd.date_range(
                        end=self.model.current_time
                        - self.model.timestep_length / substeps,
                        periods=(n_timesteps + 1)
                        * substeps,  # +1 because we prepend the discharge above
                        freq=self.model.timestep_length / substeps,
                        inclusive="right",
                    ),
                    "y": self.model.data.grid.lat,
                    "x": self.model.data.grid.lon,
                },
                dims=["time", "y", "x"],
                name="discharge",
            )
        else:
            substeps = 24  # when setting 0 it doesn't matter so much how many substeps. 24 is a reasonable default.
            n_timesteps = (event["end_time"] - event["start_time"]).days
            time = pd.date_range(
                end=self.model.current_time - self.model.timestep_length / substeps,
                periods=(n_timesteps + 1)
                * substeps,  # +1 because we prepend the discharge above
                freq=self.model.timestep_length / substeps,
                inclusive="right",
            )
            discharge_grid = np.zeros(
                shape=(len(time), *self.model.data.grid.mask.shape)
            )
            discharge_grid = xr.DataArray(
                data=discharge_grid,
                coords={
                    "time": time,
                    "y": self.model.data.grid.lat,
                    "x": self.model.data.grid.lon,
                },
                dims=["time", "y", "x"],
                name="discharge",
            )

        discharge_grid = xr.Dataset({"discharge": discharge_grid})

        from pyproj import CRS

        # Convert the WKT string to a pyproj CRS object
        crs_obj = CRS.from_wkt(self.model.data.grid.crs)

        # Now you can safely call to_proj4() on the CRS object
        discharge_grid.raster.set_crs(crs_obj.to_proj4())
        tstart = start_time
        tend = discharge_grid.time[-1] + pd.Timedelta(
            self.model.timestep_length / substeps
        )
        discharge_grid = discharge_grid.sel(time=slice(tstart, tend))

        sfincs_precipitation = event.get("precipitation", None)

        if sfincs_precipitation is None:
            sfincs_precipitation = (
                xr.open_dataset(
                    self.model.files["forcing"]["climate/pr_hourly"], engine="zarr"
                ).rename(pr_hourly="precip")["precip"]
                * 3600
            )  # convert from kg/m2/s to mm/h for
            sfincs_precipitation.raster.set_crs(
                4326
            )  # TODO: Remove when this is added to hydromt_sfincs
            sfincs_precipitation = sfincs_precipitation.rio.set_crs(4326)

        event_name = self.get_event_name(event)

        update_sfincs_model_forcing(
            model_root=self.sfincs_model_root(event_name),
            simulation_root=self.sfincs_simulation_root(event_name),
            current_event={
                "tstart": self.to_sfincs_datetime(tstart),
                "tend": self.to_sfincs_datetime(tend.dt).item(),
            },
            forcing_method="precipitation",
            discharge_grid=discharge_grid,
            precipitation_grid=sfincs_precipitation,
            data_catalogs=self.data_catalogs,
            uparea_discharge_grid=xr.open_dataset(
                self.model.files["grid"]["routing/kinematic/upstream_area"],
                engine="zarr",
            ),  # .isel(band=0),
        )
        return None

    def run_single_event(self, event, start_time, return_period=None):
        self.setup(event, force_overwrite=False)
        self.set_forcing(event, start_time)
        self.model.logger.info(f"Running SFINCS for {self.model.current_time}...")
        event_name = self.get_event_name(event)
        run_sfincs_simulation(
            simulation_root=self.sfincs_simulation_root(event_name),
            model_root=self.sfincs_model_root(event_name),
        )
        flood_map = read_flood_map(
            model_root=self.sfincs_model_root(event_name),
            simulation_root=self.sfincs_simulation_root(event_name),
            return_period=return_period,
        )  # xc, yc is for x and y in rotated grid`DD`
        damages = self.flood(
            flood_map=flood_map,
            simulation_root=self.sfincs_simulation_root(event_name),
            return_period=return_period,
        )
        return damages

    def scale_event(self, event, scale_factor):
        scaled_event = event.copy()
        sfincs_precipitation = (
            xr.open_dataset(
                self.model.files["forcing"]["climate/pr_hourly"], engine="zarr"
            ).rename(pr_hourly="precip")["precip"]
            * 3600
            * scale_factor
        )  # convert from kg/m2/s to mm/h for
        sfincs_precipitation.raster.set_crs(
            4326
        )  # TODO: Remove when this is added to hydromt_sfincs
        sfincs_precipitation = sfincs_precipitation.rio.set_crs(4326)

        scaled_event["precipitation"] = sfincs_precipitation
        return scaled_event

    def get_return_period_maps(self, config_fn="sfincs.yml", force_overwrite=True):
        # close the zarr store
        self.model.reporter.variables["discharge_daily"].close()

        model_root = self.sfincs_model_root("entire_region")
        if force_overwrite or not (model_root / "sfincs.inp").exists():
            build_sfincs(
                config_fn=str(config_fn),
                model_root=model_root,
                data_catalogs=self.data_catalogs,
                region=gpd.read_file(self.model.files["geoms"]["region"]),
                discharge_ds=self.discharge_spinup_ds,
                uparea_ds=self.uparea_ds,
                uparea_name="data",
                discharge_name="discharge_daily",
                river_method="default",
                depth_calculation="power_law",
                mask_flood_plains=True,
            )
        estimate_discharge_for_return_periods(
            model_root,
            discharge_ds=self.discharge_ds,
            data_catalogs=self.data_catalogs,
            discharge_ds_varname="discharge_daily",
        )
        run_sfincs_for_return_periods(
            model_root=model_root, return_periods=[2, 100, 1000]
        )

        # and re-open afterwards
        self.model.reporter.variables["discharge_daily"] = zarr.ZipStore(
            self.model.config["report_hydrology"]["discharge_daily"]["path"], mode="a"
        )

    def run(self, event):
        start_time = event["start_time"]

        if self.model.config["hazards"]["floods"]["flood_risk"]:
            print("config settings are read")
            scale_factors = pd.read_parquet(
                self.model.files["table"]["hydrodynamics/risk_scaling_factors"]
            )
            scale_factors["return_period"] = 1 / scale_factors["exceedance_probability"]
            damages_list = []
            return_periods_list = []
            exceedence_probabilities_list = []
            for index, row in scale_factors.iterrows():
                return_period = row["return_period"]
                exceedence_probability = row["exceedance_probability"]
                scale_factor = row["scaling_factor"]
                scaled_event = self.scale_event(event, scale_factor)
                damages = self.run_single_event(scaled_event, start_time, return_period)

                damages_list.append(damages)
                return_periods_list.append(return_period)
                exceedence_probabilities_list.append(exceedence_probability)

            print(damages_list)
            print(return_periods_list)
            print(exceedence_probabilities_list)

            plt.plot(return_periods_list, damages_list)
            plt.xlabel("Return period")
            plt.ylabel("Flood damages [euro]")
            plt.title("Damages per return period")
            plt.show()

            inverted_damage_list = damages_list[::-1]
            inverted_exceedence_probabilities_list = exceedence_probabilities_list[::-1]

            expected_annual_damage = np.trapz(
                y=inverted_damage_list, x=inverted_exceedence_probabilities_list
            )  # np.trapezoid or np.trapz -> depends on np version
            print(f"exptected annual damage is: {expected_annual_damage}")

        else:
            self.run_single_event(event, start_time)

    def flood(self, simulation_root, flood_map, return_period=None):
        damages = self.model.agents.households.flood(
            simulation_root=simulation_root,
            flood_map=flood_map,
            return_period=return_period,
        )
        return damages

    def save_discharge(self):
        self.discharge_per_timestep.append(
            self.model.data.grid.discharge_substep
        )  # this is a deque, so it will automatically remove the oldest discharge

    @property
    def discharge_spinup_ds(self):
        discharge_path = self.model.config["report_hydrology"]["discharge_daily"][
            "path"
        ].replace(self.model.run_name, "spinup")
        return xr.open_dataset(discharge_path, engine="zarr")

    @property
    def uparea_ds(self):
        uparea_path = self.model.files["grid"]["routing/kinematic/upstream_area"]
        return xr.open_dataset(uparea_path, engine="zarr")
