import geopandas as gpd
import pandas as pd
from census import Census
import os
from tqdm import tqdm

def get_us_counties():
    gdf = gpd.read_file(f"https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip")
    # remove counties in Alaska, American Samoa, Guam, Northern Marianas, Puerto Rico, and Virgin Islands
    gdf = gdf[~gdf['STATEFP'].isin(['02', '60', '66', '69', '72', '78'])]
    return gdf

fips2state = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
        "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY",  # dictionary mapping FIPS code to state abbreviation
    }

def analyze_census_data():
    c = Census("93c3297165ad8b5b6c81e0ed9e2e44a38e56224f")

    acs5_fields = (
        'B01003_001E',  # Total population
        'B25001_001E',  # Total housing units
        'B19013_001E',  # Median household income
        'B25024_002E', 'B25024_003E',  # 1-unit detached and attached structures
        'B25034_010E', 'B25034_011E',  # Structures built 2010 or later
        'B25040_002E', 'B25040_003E', 'B25040_004E',  # House heating fuel (gas, electricity, fuel oil)
        'B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E', 'B01001_024E', 'B01001_025E',  # Population 65 years and over
        'B18101_004E', 'B18101_007E', 'B18101_010E', 'B18101_023E', 'B18101_026E', 'B18101_029E',  # Disability status for 65 years and over
        'B16004_001E', 'B16004_003E',  # English speaking ability
        'B08201_002E',  # No vehicle available
        'B28002_004E', 'B28002_012E',  # Broadband internet and cellular data plan
        'C17002_002E', 'C17002_003E'  # Poverty count
    )

    
    county = c.acs5.state_county(
        fields = acs5_fields,
        state_fips = '*',
        county_fips = '*',
        year = 2022
    )
    county_df = pd.DataFrame(county)

    county_df["GEOID"] = county_df["state"] + county_df["county"]
    county_df["GEOID"] = county_df["GEOID"].astype(str)

    county_df['poverty_count'] = county_df['C17002_002E'] + county_df['C17002_003E'] 
    county_df['poverty_rate'] = county_df['poverty_count'] / county_df['B01003_001E']
    county_df['elderly_population'] = county_df['B01001_020E'] + county_df['B01001_021E'] + county_df['B01001_022E'] + county_df['B01001_023E'] + county_df['B01001_024E'] + county_df['B01001_025E']
    county_df['elderly_population_rate'] = county_df['elderly_population'] / county_df['B01003_001E']
    county_df['single_unit_housing_rate'] = (county_df['B25024_002E'] + county_df['B25024_003E']) / county_df['B25001_001E']
    county_df['new_housing_rate'] = (county_df['B25034_010E'] + county_df['B25034_011E']) / county_df['B25001_001E']
    county_df['no_vehicle_rate'] = county_df['B08201_002E'] / county_df['B25001_001E']
    county_df['internet_access_rate'] = (county_df['B28002_004E'] + county_df['B28002_012E']) / county_df['B25001_001E']

    # county_df['B19013_001E'] is null if set to -666666666
    county_df['B19013_001E'] = county_df['B19013_001E'].replace(-666666666, pd.NA)

    # Create a summary dataframe with key metrics
    metrics_df = county_df[['GEOID', 'B01003_001E', 'poverty_count', 'poverty_rate', 'B25001_001E', 'elderly_population_rate', 
                                    'single_unit_housing_rate', 'new_housing_rate', 
                                    'no_vehicle_rate', 'internet_access_rate', 'B19013_001E']]
    
    rename_dict = {
        'B01003_001E': 'Total Population',
        'poverty_count': 'Poverty Count',
        'poverty_rate': 'Poverty Rate',
        'B25001_001E': 'Total Housing Units',
        'elderly_population_rate': 'Elderly Population Rate',
        'single_unit_housing_rate': 'Single Unit Housing Rate',
        'new_housing_rate': 'New Housing Rate',
        'no_vehicle_rate': 'No Vehicle Rate',
        'internet_access_rate': 'Internet Access Rate',
        'B19013_001E': 'Median Household Income'
    }

    metrics_df = metrics_df.rename(columns=rename_dict)

    # save the metrics dataframe
    metrics_df.to_csv("../data/county_metrics.csv", index=False)

def extract_cross_model_data():
    all_counties = get_us_counties()
    grid_cells_gdf = gpd.read_file('../data/GridCellsShapefile/GridCells.shp')
    all_counties = all_counties.to_crs(crs=grid_cells_gdf.crs)
    
    # for each county, extract the cross model that intersects with the county
    for _, county in tqdm(all_counties.iterrows()):
        county_name = county['NAME']
        county_geom = county['geometry']
        state_fips = county["STATEFP"]
        county_state = fips2state[state_fips]
        cross_models = grid_cells_gdf[grid_cells_gdf.intersects(county_geom)]
        folder = f"../data/cross_models/{county_state}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        cross_models['GEOID'] = county['GEOID']
        cross_models.to_csv(f"../data/cross_models/{county_state}/{county_name}.csv", index=False)

# run this function to get socio-economic metrics for each county
analyze_census_data()

# run this function to extract cross models for each county
extract_cross_model_data()

    