import os
import math
import rasterio
import hydromt
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from preconfig import INPUT, ORIGINAL_DATA, PREPROCESSING_FOLDER
from hydromt_geb.workflows import get_farm_distribution

data_catalog = hydromt.DataCatalog(os.path.join(ORIGINAL_DATA, "data_catalog.yml"))


def main():
    SIZE_CLASSES_BOUNDARIES = {
        "< 1 Ha": (0, 10000),
        "1 - 2 Ha": (10000, 20000),
        "2 - 5 Ha": (20000, 50000),
        "5 - 10 Ha": (50000, 100000),
        "10 - 20 Ha": (100000, 200000),
        "20 - 50 Ha": (200000, 500000),
        "50 - 100 Ha": (500000, 1000000),
        "100 - 200 Ha": (1000000, 2000000),
        "200 - 500 Ha": (2000000, 5000000),
        "500 - 1000 Ha": (5000000, 10000000),
        "> 1000 Ha": (10000000, 20000000),
    }

    with rasterio.open(
        os.path.join(INPUT, "landsurface", "full_region_cultivated_land.tif"), "r"
    ) as src:
        cultivated_land = src.read(1)

    with rasterio.open(
        os.path.join(INPUT, "areamaps", "region_subgrid.tif"), "r"
    ) as src:
        regions_grid = src.read(1)

    with rasterio.open(
        os.path.join(INPUT, "areamaps", "region_cell_area_subgrid.tif"), "r"
    ) as src:
        cell_area = src.read(1)

    regions_shapes = gpd.read_file(os.path.join(INPUT, "areamaps", "regions.geojson"))

    farm_sizes_per_country = (
        data_catalog.get_dataframe("lowder_farm_sizes")
        .dropna(subset=["Total"], axis=0)
        .drop(["empty", "income class"], axis=1)
    )
    farm_sizes_per_country["Country"] = farm_sizes_per_country["Country"].ffill()
    farm_sizes_per_country["Census Year"] = farm_sizes_per_country["Country"].ffill()

    all_agents = []
    for _, region in regions_shapes.iterrows():
        UID = region["UID"]
        country = region["NAME_0"]
        print(f"Processing region {UID} in {country}")
        cultivated_land_region = cultivated_land[regions_grid == UID]
        total_cultivated_land_area_lu = cell_area[
            (regions_grid == UID) & (cultivated_land == True)
        ].sum()
        average_cell_area_region = cell_area[
            (regions_grid == UID) & (cultivated_land == True)
        ].mean()

        country_farm_sizes = farm_sizes_per_country.loc[
            (farm_sizes_per_country["Country"] == country)
        ].drop(["Country", "Census Year", "Total"], axis=1)
        assert (
            len(country_farm_sizes) == 2
        ), f"Found {len(country_farm_sizes) / 2} country_farm_sizes for {country}"

        n_holdings = (
            country_farm_sizes.loc[
                country_farm_sizes["Holdings/ agricultural area"] == "Holdings"
            ]
            .iloc[0]
            .drop(["Holdings/ agricultural area"])
            .replace("..", "0")
            .astype(np.int64)
        )
        agricultural_area_db_ha = (
            country_farm_sizes.loc[
                country_farm_sizes["Holdings/ agricultural area"]
                == "Agricultural area (Ha) "
            ]
            .iloc[0]
            .drop(["Holdings/ agricultural area"])
            .replace("..", "0")
            .astype(np.int64)
        )
        agricultural_area_db = agricultural_area_db_ha * 10000
        avg_size_class = agricultural_area_db / n_holdings

        total_cultivated_land_area_db = agricultural_area_db.sum()

        n_cells_per_size_class = pd.Series(0, index=n_holdings.index)

        for size_class in agricultural_area_db.index:
            if n_holdings[size_class] > 0:
                n_holdings[size_class] = n_holdings[size_class] * (
                    total_cultivated_land_area_lu / total_cultivated_land_area_db
                )
                n_cells_per_size_class.loc[size_class] = (
                    n_holdings[size_class]
                    * avg_size_class[size_class]
                    / average_cell_area_region
                )
                assert not np.isnan(n_cells_per_size_class.loc[size_class])

        assert math.isclose(cultivated_land_region.sum(), n_cells_per_size_class.sum())

        whole_cells_per_size_class = (n_cells_per_size_class // 1).astype(int)
        leftover_cells_per_size_class = n_cells_per_size_class % 1
        whole_cells = whole_cells_per_size_class.sum()
        n_missing_cells = cultivated_land_region.sum() - whole_cells
        assert n_missing_cells <= len(agricultural_area_db)

        index = list(
            zip(leftover_cells_per_size_class.index, leftover_cells_per_size_class % 1)
        )
        n_cells_to_add = sorted(index, key=lambda x: x[1], reverse=True)[
            :n_missing_cells
        ]
        whole_cells_per_size_class.loc[[p[0] for p in n_cells_to_add]] += 1

        assert whole_cells_per_size_class.sum() == cultivated_land_region.sum()

        region_agents = []
        for size_class in whole_cells_per_size_class.index:
            # if no cells for this size class, just continue
            if whole_cells_per_size_class.loc[size_class] == 0:
                continue

            min_size_m2, max_size_m2 = SIZE_CLASSES_BOUNDARIES[size_class]

            min_size_cells = int(min_size_m2 / average_cell_area_region)
            min_size_cells = max(
                min_size_cells, 1
            )  # farm can never be smaller than one cell
            max_size_cells = (
                int(max_size_m2 / average_cell_area_region) - 1
            )  # otherwise they overlap with next size class
            mean_cells_per_agent = int(
                avg_size_class[size_class] / average_cell_area_region
            )

            if (
                mean_cells_per_agent < min_size_cells
                or mean_cells_per_agent > max_size_cells
            ):  # there must be an error in the data, thus assume centred
                mean_cells_per_agent = (min_size_cells + max_size_cells) // 2

            number_of_agents_size_class = round(n_holdings[size_class])
            # if there is agricultural land, but there are no agents rounded down, we assume there is one agent
            if (
                number_of_agents_size_class == 0
                and whole_cells_per_size_class[size_class] > 0
            ):
                number_of_agents_size_class = 1

            population = pd.DataFrame(index=range(number_of_agents_size_class))

            offset = (
                whole_cells_per_size_class[size_class]
                - number_of_agents_size_class * mean_cells_per_agent
            )

            n_farms_size_class, farm_sizes_size_class = get_farm_distribution(
                number_of_agents_size_class,
                min_size_cells,
                max_size_cells,
                mean_cells_per_agent,
                offset,
            )
            assert n_farms_size_class.sum() == number_of_agents_size_class
            assert (farm_sizes_size_class > 0).all()
            assert (
                n_farms_size_class * farm_sizes_size_class
            ).sum() == whole_cells_per_size_class[size_class]
            farm_sizes = farm_sizes_size_class.repeat(n_farms_size_class)
            np.random.shuffle(farm_sizes)
            population["area_n_cells"] = farm_sizes
            region_agents.append(population)

            assert (
                population["area_n_cells"].sum()
                == whole_cells_per_size_class[size_class]
            )

        region_agents = pd.concat(region_agents, ignore_index=True)
        region_agents["region_id"] = UID
        all_agents.append(region_agents)

    all_agents = pd.concat(all_agents, ignore_index=True)

    folder = os.path.join(PREPROCESSING_FOLDER, "agents", "farmers")
    os.makedirs(folder, exist_ok=True)
    all_agents.to_csv(os.path.join(folder, "farmers.csv"), index=True)


if __name__ == "__main__":
    main()
