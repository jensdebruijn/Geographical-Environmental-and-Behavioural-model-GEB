import os

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from methods import create_cell_area_map

from config import INPUT

base = importr('base')
ipfr = importr('ipfr')
dplyr = importr('dplyr')
rtibble = importr('tibble')

base.set_seed(42)

def df_to_tibble(df):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataframe = ro.conversion.py2rpy(df)
    return rtibble.as_tibble(r_dataframe)

def tibble_to_df(tibble):
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(tibble)

SIZE_CLASSES = (
    'Below 0.5',
    '0.5-1.0',
    '1.0-2.0',
    '2.0-3.0',
    '3.0-4.0',
    '4.0-5.0',
    '5.0-7.5',
    '7.5-10.0',
    '10.0-20.0',
    '20.0 & ABOVE',
)

with rasterio.open(os.path.join(INPUT, 'tehsils.tif'), 'r') as src:
    tehsils_tif = src.read(1)
    cell_area = create_cell_area_map(src.profile, write_to_disk=True)

tehsils_shape = gpd.read_file(os.path.join(INPUT, 'tehsils.geojson')).set_index(['State', 'District', 'Tehsil'])
avg_farm_size = pd.read_excel(os.path.join(INPUT, 'census', 'avg_farm_size.xlsx'), index_col=(0, 1, 2))
crop_data = pd.read_excel(os.path.join(INPUT, 'census', 'crop_data.xlsx'), index_col=(0, 1, 2, 3))
for (state, district, tehsil), tehsil_crop_data in crop_data.groupby(level=[0, 1, 2]):
    tehsil_farm_size = avg_farm_size.loc[(state, district, tehsil)]
    farms_per_size_class = tehsil_crop_data.droplevel([0, 1, 2]).sum(axis=1)

    assert (tehsil_farm_size.index == farms_per_size_class.index).all()

    area_per_size_class = tehsil_farm_size * farms_per_size_class
    census_farm_area = area_per_size_class.sum()

    tehsil_ID = tehsils_shape.at[(state, district, tehsil), 'ID']
    tehsil_area = cell_area[tehsils_tif == tehsil_ID].sum()

    print(state, district, tehsil, census_farm_area, tehsil_area, census_farm_area/tehsil_area)

    continue

exit()

size_class_convert = {
    "Below 0.5": 0,
    "0.5-1.0": 1,
    "1.0-2.0": 2,
    "2.0-3.0": 3,
    "3.0-4.0": 4,
    "4.0-5.0": 5,
    "5.0-7.5": 6,
    "7.5-10.0": 7,
    "10.0-20.0": 8,
    "20.0 & ABOVE": 9,
}

def assign_size_classes(survey):
    size_classes = (
        ('Below 0.5', 0.5),
        ('0.5-1.0', 1),
        ('1.0-2.0', 2),
        ('2.0-3.0', 3),
        ('3.0-4.0', 4),
        ('4.0-5.0', 5),
        ('5.0-7.5', 7.5),
        ('7.5-10.0', 10),
        ('10.0-20.0', 20),
        ('20.0 & ABOVE', np.inf),
    )
    for idx, household in survey.iterrows():
        area = household['area owned & cultivated']
        for size_class_name, size in size_classes:
            if area < size:
                survey.loc[idx, 'size_class'] = size_class_name
                break
    return survey

survey_data = pd.read_csv(os.path.join(INPUT, 'agents', 'IHDS_I.csv'))
survey_data = assign_size_classes(survey_data)
survey_data = survey_data[~survey_data['Kharif: Crop: Name'].isnull()]
survey_data['is_irrigated'] = survey_data['Kharif: Crop: Irrigation'].map({'Yes': 'irr', 'No': 'rain'})

crop_convert = pd.read_excel(os.path.join(INPUT, 'crops', 'crops.xlsx')).set_index('IHDS')['CENSUS'].to_dict()
survey_data['crop'] = survey_data['Kharif: Crop: Name'].map(crop_convert)
survey_data = survey_data[~survey_data['crop'].isnull()]
survey_data['crop'] = survey_data['crop'] + '_' + survey_data['is_irrigated']
# survey_data['size_class'] = survey_data['size_class'].map(size_class_convert)

folder = os.path.join(INPUT, 'agents', 'population')
os.makedirs(folder, exist_ok=True)

for (state, district, tehsil, size_class), crop_frequencies in crop_data.groupby(crop_data.index):
    print(state, district, tehsil, size_class)
    # survey_data_size_class = survey_data[survey_data['size_class'] == size_class]
    print('split by size class')
    survey_data_size_class = survey_data
    survey_data_size_class = survey_data_size_class.reset_index(drop=True)

    survey = df_to_tibble(survey_data_size_class[['crop', 'weight']])
    targets = ro.ListVector({
        'crop': df_to_tibble(pd.DataFrame([
            crop_frequencies.iloc[0].tolist(),
        ], columns=list(crop_frequencies.columns)))
    })

    result = ipfr.ipu(survey, targets)

    base.set_seed(42)

    population = tibble_to_df(ipfr.synthesize(result.rx2('weight_tbl'))).drop(['new_id', 'crop'], axis=1)
    population['id'] -= 1  # R is 1-indexed and Python is 0-indexed.
    population = population.merge(survey_data_size_class, how='left', left_on='id', right_index=True)
    population = population.drop(['id', 'weight', 'State code', 'District code', 'PSU: village/neighborhood code', 'Household ID', 'Split household ID', 'Full household ID'], axis=1)

    fp = os.path.join(folder, f"{state}_{district}_{tehsil}_{size_class}.csv")
    population.to_csv(fp, index=False)