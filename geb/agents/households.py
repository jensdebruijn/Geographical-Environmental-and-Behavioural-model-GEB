import numpy as np
import geopandas as gpd
import calendar
from .general import AgentArray, downscale_volume, AgentBaseClass
from ..hydrology.landcover import SEALED
import pandas as pd
from os.path import join
from damagescanner.core import object_scanner
import json
import xarray as xr
import rioxarray
from rasterio.features import shapes
import rasterio
from shapely.geometry import shape

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass


def from_landuse_raster_to_polygon(rasterdata, landuse_category):
    """
    Convert raster data into separate GeoDataFrames for specified land use values.

    Parameters:
    - landuse: An xarray DataArray or similar with land use data and 'x' and 'y' coordinates.
    - values_to_extract: List of integer values to extract (e.g., [0, 1] for forest and agriculture).

    Returns:
    - Geodataframe
    """
    data = rasterdata["data"].values
    data = data.astype(np.uint8)

    y_coords = rasterdata.coords["y"].values
    x_coords = rasterdata.coords["x"].values

    transform = rasterio.transform.from_origin(
        x_coords[0],
        y_coords[0],
        abs(x_coords[1] - x_coords[0]),
        abs(y_coords[1] - y_coords[0]),
    )

    mask = data == landuse_category

    shapes_gen = shapes(data, mask=mask, transform=transform)

    polygons = []
    for geom, value in shapes_gen:
        if value == landuse_category:
            polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame(
        {"value": [landuse_category] * len(polygons), "geometry": polygons}
    )
    gdf.set_crs(epsg=4326, inplace=True)

    return gdf


class Households(AgentBaseClass):
    def __init__(self, model, agents, reduncancy: float) -> None:
        self.model = model
        self.agents = agents
        self.reduncancy = reduncancy
        self.config = (
            self.model.config["agent_settings"]["households"]
            if "households" in self.model.config["agent_settings"]
            else {}
        )
        # Load buildings
        self.buildings = gpd.read_file(self.model.files["geoms"]["assets/buildings"])
        self.buildings["object_type"] = "building_structure"
        self.buildings_centroid = gpd.GeoDataFrame(geometry=self.buildings.centroid)
        self.buildings_centroid["object_type"] = "building_content"

        # Load roads
        self.roads = gpd.read_file(self.model.files["geoms"]["assets/roads"])
        self.roads = self.roads.rename(columns={"highway": "object_type"})

        # Load rail
        self.rail = gpd.read_file(self.model.files["geoms"]["assets/rails"])
        self.rail["object_type"] = "rail"

        # Load landuse and make turn into polygons
        self.landuse = xr.open_zarr(
            self.model.files["region_subgrid"][
                "landsurface/full_region_land_use_classes"
            ]
        )
        self.forest = from_landuse_raster_to_polygon(self.landuse, 0)
        self.forest["object_type"] = "forest"

        self.agriculture = from_landuse_raster_to_polygon(self.landuse, 1)
        self.agriculture["object_type"] = "agriculture"

        # Load maximum damages
        with open(
            model.files["dict"]["damage_parameters/flood/rail/main/maximum_damage"], "r"
        ) as f:
            self.max_dam_rail = json.load(f)
        self.max_dam_rail = float(self.max_dam_rail["maximum_damage"])
        self.rail["maximum_damage"] = self.max_dam_rail

        self.max_dam_road = {}
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/maximum_damage"),
            (
                "unclassified",
                "damage_parameters/flood/road/unclassified/maximum_damage",
            ),
            ("tertiary", "damage_parameters/flood/road/tertiary/maximum_damage"),
            ("primary", "damage_parameters/flood/road/primary/maximum_damage"),
            (
                "primary_link",
                "damage_parameters/flood/road/primary_link/maximum_damage",
            ),
            ("secondary", "damage_parameters/flood/road/secondary/maximum_damage"),
            (
                "secondary_link",
                "damage_parameters/flood/road/secondary_link/maximum_damage",
            ),
            ("motorway", "damage_parameters/flood/road/motorway/maximum_damage"),
            (
                "motorway_link",
                "damage_parameters/flood/road/motorway_link/maximum_damage",
            ),
            ("trunk", "damage_parameters/flood/road/trunk/maximum_damage"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/maximum_damage"),
        ]

        for road_type, path in road_types:
            with open(model.files["dict"][path], "r") as f:
                max_damage = json.load(f)
            self.max_dam_road[road_type] = max_damage["maximum_damage"]

        self.roads["maximum_damage"] = self.roads["object_type"].map(self.max_dam_road)

        with open(
            model.files["dict"][
                "damage_parameters/flood/land_use/forest/maximum_damage"
            ],
            "r",
        ) as f:
            self.max_dam_forest = json.load(f)
        self.max_dam_forest = float(self.max_dam_forest["maximum_damage"])
        self.forest["maximum_damage"] = self.max_dam_forest

        with open(
            model.files["dict"][
                "damage_parameters/flood/land_use/agriculture/maximum_damage"
            ],
            "r",
        ) as f:
            self.max_dam_agriculture = json.load(f)
        self.max_dam_agriculture = float(self.max_dam_agriculture["maximum_damage"])
        self.agriculture["maximum_damage"] = self.max_dam_agriculture

        # Load vulnerability curves
        self.road_curves = []
        road_types = [
            ("residential", "damage_parameters/flood/road/residential/curve"),
            ("unclassified", "damage_parameters/flood/road/unclassified/curve"),
            ("tertiary", "damage_parameters/flood/road/tertiary/curve"),
            ("primary", "damage_parameters/flood/road/primary/curve"),
            ("primary_link", "damage_parameters/flood/road/primary_link/curve"),
            ("secondary", "damage_parameters/flood/road/secondary/curve"),
            ("secondary_link", "damage_parameters/flood/road/secondary_link/curve"),
            ("motorway", "damage_parameters/flood/road/motorway/curve"),
            ("motorway_link", "damage_parameters/flood/road/motorway_link/curve"),
            ("trunk", "damage_parameters/flood/road/trunk/curve"),
            ("trunk_link", "damage_parameters/flood/road/trunk_link/curve"),
        ]

        severity_column = None
        for road_type, path in road_types:
            df = pd.read_parquet(self.model.files["table"][path])

            if severity_column is None:
                severity_column = df["severity"]

            df = df.rename(columns={"damage_ratio": road_type})

            self.road_curves.append(df[[road_type]])
        self.road_curves = pd.concat([severity_column] + self.road_curves, axis=1)
        self.road_curves.set_index("severity", inplace=True)

        self.forest_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/land_use/forest/curve"]
        )
        self.forest_curve.set_index("severity", inplace=True)
        self.forest_curve = self.forest_curve.rename(columns={"damage_ratio": "forest"})
        self.agriculture_curve = pd.read_parquet(
            self.model.files["table"][
                "damage_parameters/flood/land_use/agriculture/curve"
            ]
        )
        self.agriculture_curve.set_index("severity", inplace=True)
        self.agriculture_curve = self.agriculture_curve.rename(
            columns={"damage_ratio": "agriculture"}
        )

        # Get vulnerability curves based on adaptation scenario
        if self.config.get("measure") not in ["dry_proofing", "wet_proofing"]:
            print("were going for the normal curves ")
            self.buildings_structure_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/normal/structure/curve"
                ]
            )
            self.buildings_structure_curve.set_index("severity", inplace=True)
            self.buildings_structure_curve = self.buildings_structure_curve.rename(
                columns={"damage_ratio": "building_structure"}
            )

            self.buildings_content_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/normal/content/curve"
                ]
            )
            self.buildings_content_curve.set_index("severity", inplace=True)
            self.buildings_content_curve = self.buildings_content_curve.rename(
                columns={"damage_ratio": "building_content"}
            )
            with open(
                model.files["dict"][
                    "damage_parameters/flood/buildings/normal/structure/maximum_damage"
                ],
                "r",
            ) as f:
                self.max_dam_buildings_structure = json.load(f)
            self.max_dam_buildings_structure = float(
                self.max_dam_buildings_structure["maximum_damage"]
            )
            self.buildings["maximum_damage"] = self.max_dam_buildings_structure
            with open(
                model.files["dict"][
                    "damage_parameters/flood/buildings/normal/content/maximum_damage"
                ],
                "r",
            ) as f:
                self.max_dam_buildings_content = json.load(f)
            self.max_dam_buildings_content = float(
                self.max_dam_buildings_content["maximum_damage"]
            )
            self.buildings_centroid["maximum_damage"] = self.max_dam_buildings_content

        if self.config.get("measure") == "dry_proofing":
            print("turning on dryproofing")
            self.buildings_structure_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/dry_proofing/structure/curve"
                ]
            )
            self.buildings_structure_curve.set_index("severity", inplace=True)
            self.buildings_structure_curve = self.buildings_structure_curve.rename(
                columns={"damage_ratio": "building_structure"}
            )

            self.buildings_content_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/dry_proofing/content/curve"
                ]
            )
            self.buildings_content_curve.set_index("severity", inplace=True)
            self.buildings_content_curve = self.buildings_content_curve.rename(
                columns={"damage_ratio": "building_content"}
            )
            print(self.buildings_structure_curve)
            print(self.buildings_content_curve)
            with open(
                model.files["dict"][
                    "damage_parameters/flood/buildings/dry_proofing/structure/maximum_damage"
                ],
                "r",
            ) as f:
                self.max_dam_buildings_structure = json.load(f)
            self.max_dam_buildings_structure = float(
                self.max_dam_buildings_structure["maximum_damage"]
            )
            self.buildings["maximum_damage"] = self.max_dam_buildings_structure
            with open(
                model.files["dict"][
                    "damage_parameters/flood/buildings/dry_proofing/content/maximum_damage"
                ],
                "r",
            ) as f:
                self.max_dam_buildings_content = json.load(f)
            self.max_dam_buildings_content = float(
                self.max_dam_buildings_content["maximum_damage"]
            )
            self.buildings_centroid["maximum_damage"] = self.max_dam_buildings_content
        if self.config.get("measure") == "wet_proofing":
            print("turning on wetproofing")
            self.buildings_structure_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/wet_proofing/structure/curve"
                ]
            )
            self.buildings_structure_curve.set_index("severity", inplace=True)
            self.buildings_structure_curve = self.buildings_structure_curve.rename(
                columns={"damage_ratio": "building_structure"}
            )
            print("builing structure curve")
            print(self.buildings_structure_curve)

            self.buildings_content_curve = pd.read_parquet(
                self.model.files["table"][
                    "damage_parameters/flood/buildings/wet_proofing/content/curve"
                ]
            )
            self.buildings_content_curve.set_index("severity", inplace=True)
            self.buildings_content_curve = self.buildings_content_curve.rename(
                columns={"damage_ratio": "building_content"}
            )

            print("building content curve")
            print(self.buildings_content_curve)
            with open(
                model.files["dict"][
                    "damage_parameters/flood/buildings/wet_proofing/structure/maximum_damage"
                ],
                "r",
            ) as f:
                self.max_dam_buildings_structure = json.load(f)
            self.max_dam_buildings_structure = float(
                self.max_dam_buildings_structure["maximum_damage"]
            )
            self.buildings["maximum_damage"] = self.max_dam_buildings_structure
            with open(
                model.files["dict"][
                    "damage_parameters/flood/buildings/wet_proofing/content/maximum_damage"
                ],
                "r",
            ) as f:
                self.max_dam_buildings_content = json.load(f)
            self.max_dam_buildings_content = float(
                self.max_dam_buildings_content["maximum_damage"]
            )
            self.buildings_centroid["maximum_damage"] = self.max_dam_buildings_content

        self.rail_curve = pd.read_parquet(
            self.model.files["table"]["damage_parameters/flood/rail/main/curve"]
        )
        self.rail_curve.set_index("severity", inplace=True)
        self.rail_curve = self.rail_curve.rename(columns={"damage_ratio": "rail"})

        super().__init__()

        water_demand, efficiency = self.update_water_demand()
        self.current_water_demand = water_demand
        self.current_efficiency = efficiency

    def initiate(self) -> None:
        locations = np.load(self.model.files["binary"]["agents/households/locations"])[
            "data"
        ]
        self.max_n = int(locations.shape[0] * (1 + self.reduncancy) + 1)

        self.locations = AgentArray(locations, max_n=self.max_n)

        sizes = np.load(self.model.files["binary"]["agents/households/sizes"])["data"]
        self.sizes = AgentArray(sizes, max_n=self.max_n)

        self.flood_depth = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=0, dtype=np.float32
        )
        self.risk_perception = AgentArray(
            n=self.n, max_n=self.max_n, fill_value=1, dtype=np.float32
        )

    def flood(self, flood_map, model_root, simulation_root, return_period=None):
        if return_period is not None:
            flood_path = join(simulation_root, f"hmax RP {int(return_period)}.tif")
        else:
            flood_path = join(simulation_root, "hmax.tif")

        print(f"using this flood map: {flood_path}")
        flood_map = rioxarray.open_rasterio(flood_path)

        # Remove rivers from flood map
        rivers_path = join(model_root, "rivers.gpkg")
        rivers = gpd.read_file(rivers_path)
        rivers.set_crs(epsg=4326, inplace=True)
        rivers_projected = rivers.to_crs(flood_map.rio.crs)
        rivers_projected["geometry"] = rivers_projected.buffer(
            rivers_projected["rivwth"] / 2
        )
        rivers_mask = flood_map.raster.geometry_mask(
            gdf=rivers_projected, all_touched=True
        )
        flood_map = flood_map.where(~rivers_mask)
        flood_map = flood_map.fillna(0)
        flood_map = flood_map.where(flood_map != 0, np.nan)

        # Clip the flood map to the region for which we want to know the damages
        region_path = "/scistor/ivm/vbl220/PhD/damages_region.gpkg"
        region = gpd.read_file(region_path)
        region_projected = region.to_crs(flood_map.rio.crs)
        flood_map_clipped = flood_map.rio.clip(
            region_projected.geometry, region_projected.crs
        )

        # If scenario is water buffer, no damages within the buffer, so clip out location of water buffers
        # if self.model.config["hazards"]["floods"]["measure"] == "waterbuffers":
        #     # waterbuffer_locations = gpd.read_file(
        #     #     "/scistor/ivm/vbl220/PhD/waterbuffer_more_info.gpkg"
        #     # )
        #     waterbuffer_locations = gpd.read_file(
        #         "/scistor/ivm/vbl220/PhD/large_water_buffer.gpkg"
        #     )
        #     waterbuffer_locations_reprojected = waterbuffer_locations.to_crs(
        #         flood_map.rio.crs
        #     )
        #     flood_map_clipped = flood_map_clipped.rio.clip(
        #         waterbuffer_locations_reprojected.geometry,
        #         waterbuffer_locations_reprojected.crs,
        #         invert=True,
        #     )

        def compute_damages_by_country(assets, curve, category_name):
            assets = assets.to_crs(flood_map_clipped.rio.crs)

            # Check for multiple geometry types
            geometry_types = assets.geometry.geom_type.unique()
            print(f"Geometry types in {category_name}: {geometry_types}")

            if "MultiPolygon" in geometry_types:
                assets = assets.explode(index_parts=False).reset_index(drop=True)

            # If only one geometry type, proceed normally
            if len(geometry_types) == 1:
                damages = object_scanner(
                    objects=assets, hazard=flood_map_clipped, curves=curve
                )
                assets["damages"] = damages
                total_damages = damages.sum()
                print(f"damages to {category_name} are: {total_damages}")

                split_assets = gpd.overlay(
                    assets,
                    gdf_filtered_countries,
                    how="intersection",
                    keep_geom_type=True,
                )
                unmatched_assets = split_assets[split_assets["COUNTRY"].isnull()]

                for country in selection_countries:
                    country_assets = split_assets[split_assets["COUNTRY"] == country]
                    if not country_assets.empty:
                        country_assets = country_assets.to_crs(
                            flood_map_clipped.rio.crs
                        )
                        country_damages = object_scanner(
                            objects=country_assets,
                            hazard=flood_map_clipped,
                            curves=curve,
                        ).sum()
                        print(
                            f"damages to {category_name} ({country}): {country_damages}"
                        )

                return total_damages

            # If multiple geometry types, split and process each separately
            if len(geometry_types) > 1:
                total_damages = 0
                for geom_type in geometry_types:
                    print(f"Processing geometry type: {geom_type}")
                    subset = assets[assets.geometry.geom_type == geom_type]

                    # Process subset as usual
                    damages = object_scanner(
                        objects=subset, hazard=flood_map_clipped, curves=curve
                    )
                    subset_damages = damages.sum()
                    total_damages += subset_damages
                    print(
                        f"damages to {category_name} ({geom_type}) are: {subset_damages}"
                    )

                    # Perform overlay for this subset
                    split_assets = gpd.overlay(
                        subset, gdf_filtered_countries, how="intersection"
                    )
                    unmatched_assets = split_assets[split_assets["COUNTRY"].isnull()]
                    print(f"Unmatched assets for {geom_type}: {unmatched_assets}")

                    for country in selection_countries:
                        country_assets = split_assets[
                            split_assets["COUNTRY"] == country
                        ]
                        if not country_assets.empty:
                            country_assets = country_assets.to_crs(
                                flood_map_clipped.rio.crs
                            )
                            country_damages = object_scanner(
                                objects=country_assets,
                                hazard=flood_map_clipped,
                                curves=curve,
                            ).sum()
                            print(
                                f"damages to {category_name} ({country}, {geom_type}): {country_damages}"
                            )
                print(
                    f"Total damages to {category_name} (all geometry types): {total_damages}"
                )
                return total_damages

        # Filter countries
        all_countries = gpd.read_file("/scistor/ivm/vbl220/PhD/Europe_merged.shp")
        selection_countries = ["Netherlands", "Belgium", "Germany"]
        gdf_filtered_countries = all_countries[
            all_countries["COUNTRY"].isin(selection_countries)
        ]
        if self.buildings.crs != flood_map.rio.crs:
            gdf_filtered_countries = gdf_filtered_countries.to_crs(flood_map.rio.crs)

        # Compute damages for each category
        total_damages_agriculture = compute_damages_by_country(
            self.agriculture, self.agriculture_curve, "agriculture"
        )
        total_damages_forest = compute_damages_by_country(
            self.forest, self.forest_curve, "forest"
        )
        total_damage_structure = compute_damages_by_country(
            self.buildings, self.buildings_structure_curve, "building structure"
        )
        total_damages_content = compute_damages_by_country(
            self.buildings_centroid, self.buildings_content_curve, "building content"
        )
        total_damages_roads = compute_damages_by_country(
            self.roads, self.road_curves, "roads"
        )
        total_damages_rail = compute_damages_by_country(
            self.rail, self.rail_curve, "rail"
        )

        # Calculate total flood damages
        total_flood_damages = (
            total_damage_structure
            + total_damages_content
            + total_damages_roads
            + total_damages_rail
            + total_damages_forest
            + total_damages_agriculture
        )
        print(f"the total flood damages are: {total_flood_damages}")
        return total_flood_damages

    def update_water_demand(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        downscale_mask = self.model.data.HRU.land_use_type != SEALED
        if self.model.use_gpu:
            downscale_mask = downscale_mask.get()
        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
        water_demand = (
            self.model.domestic_water_demand_ds.sel(
                time=self.model.current_time, method="ffill", tolerance="366D"
            ).domestic_water_demand
            * 1_000_000
            / days_in_year
        )
        water_demand = downscale_volume(
            self.model.domestic_water_demand_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_demand = cp.array(water_demand)
        water_demand = self.model.data.HRU.M3toM(water_demand)

        water_consumption = (
            self.model.domestic_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).domestic_water_consumption
            * 1_000_000
            / days_in_year
        )
        water_consumption = downscale_volume(
            self.model.domestic_water_consumption_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.model.data.HRU.land_use_ratio,
        )
        if self.model.use_gpu:
            water_consumption = cp.array(water_consumption)
        water_consumption = self.model.data.HRU.M3toM(water_consumption)

        efficiency = np.divide(
            water_consumption,
            water_demand,
            out=np.zeros_like(water_consumption, dtype=float),
            where=water_demand != 0,
        )

        efficiency = self.model.data.to_grid(HRU_data=efficiency, fn="max")

        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        self.last_water_demand_update = self.model.current_time
        return water_demand, efficiency

    def water_demand(self):
        if (
            np.datetime64(self.model.current_time, "ns")
            in self.model.domestic_water_consumption_ds.time
        ):
            water_demand, efficiency = self.update_water_demand()
            self.current_water_demand = water_demand
            self.current_efficiency = efficiency

        assert (self.model.current_time - self.last_water_demand_update).days < 366, (
            "Water demand has not been updated for over a year. "
            "Please check the household water demand datasets."
        )
        return self.current_water_demand, self.current_efficiency

    def step(self) -> None:
        self.risk_perception *= self.risk_perception
        return None

    @property
    def n(self):
        return self.locations.shape[0]
