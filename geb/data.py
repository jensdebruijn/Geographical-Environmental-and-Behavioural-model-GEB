import os
from datetime import date, datetime
import numpy as np
import pandas as pd
import json
from pathlib import Path


class DateIndex:
    def __init__(self, dates):
        self.dates = np.array(dates)

    def get(self, date):
        # find first date where date is larger or equal to date in self.dates
        return np.searchsorted(self.dates, date, side="right") - 1

    def __len__(self):
        return self.dates.size


def load_regional_crop_data_from_dict(
    model, name
) -> tuple[dict[dict[date, int]], dict[str, np.ndarray]]:
    """Load crop prices per state from the input data and return a dictionary of states containing 2D array of prices.

    Returns:
        date_index: Dictionary of states containing a dictionary of dates and their index in the 2D array.
        crop_prices: Dictionary of states containing a 2D array of crop prices. First index is for date, second index is for crop.
    """

    with open(model.model_structure["dict"][name], "r") as f:
        timedata = json.load(f)

    dates = parse_dates(timedata["time"])
    date_index = DateIndex(dates)

    data = timedata["data"]

    d = np.full(
        (len(date_index), len(model.regions), len(data["0"])), np.nan, dtype=np.float32
    )  # all lengths should be the same, so just taking data from region 0.
    for region_id, region_data in data.items():
        for ID, region_crop_data in region_data.items():
            d[:, int(region_id), int(ID)] = region_crop_data

    assert not np.isnan(d).any()
    return date_index, d


def load_crop_variables(model_structure) -> dict[np.ndarray]:
    """Read csv-file of values for crop water depletion.

    Returns:
        yield_factors: dictonary with np.ndarray of values per crop for each variable.
    """
    with open(model_structure["dict"]["crops/crop_variables"], "r") as f:
        crop_variables = json.load(f)
    return pd.DataFrame.from_dict(crop_variables, orient="index")


def parse_dates(date_strings, date_formats=["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y"]):
    for date_format in date_formats:
        try:
            return [datetime.strptime(str(d), date_format) for d in date_strings]
        except ValueError:
            pass
    else:
        raise ValueError(
            "No valid date format found for date strings: {}".format(date_strings[0])
        )


def load_crop_ids(model_structure):
    with open(model_structure["dict"]["crops/crop_ids"], "r") as f:
        crop_ids = json.load(f)
    # convert keys to int
    crop_ids = {int(key): value for key, value in crop_ids.items()}
    return crop_ids


def load_economic_data(fp: str) -> tuple[DateIndex, dict[int, np.ndarray]]:
    with open(fp, "r") as f:
        data = json.load(f)
    dates = parse_dates(data["time"])
    date_index = DateIndex(dates)
    d = {int(region_id): values for region_id, values in data["data"].items()}
    return (date_index, d)


# def load_sprinkler_prices(self, inflation_rates_per_year):
#     sprinkler_price_2008 = self.model.config['agent_settings']['expected_utility']['adaptation_sprinkler']['adaptation_cost']
#     #upkeep_price_2008_m2 = 3000 / 10_000  # ha to m2
#     # create dictory with prices for well_prices per year by applying inflation rates
#     sprinkler_prices = {2008: sprinkler_price_2008}
#     for year in range(2009, 2022):
#         sprinkler_prices[year] = sprinkler_prices[year-1] * inflation_rates_per_year[year]
#     for year in range(2007, 1960, -1):
#         sprinkler_prices[year] = sprinkler_prices[year+1] / inflation_rates_per_year[year+1]
#     return sprinkler_prices
