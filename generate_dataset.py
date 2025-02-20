import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm

# Force geopandas not to use pygeos.
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

# External modules
from data_retrieval import retrieve_data_from_location
from query_llm import QueryLLM
import utils
import visualization
import ocr
import oracle


class BenchmarkDatasetGenerator:
    """
    Generates benchmark dataset samples by retrieving climate data tables, visualizing them,
    merging outputs, and finally querying an oracle for answers.

    The retrieval is divided into distinct cases:
      (1) Single data table.
      (2) Two data tables (if only one of 'climate_variable2', 'location2' or 'time_frame2' exists).
      (3) Four data tables:
           (3.1) When both 'time_frame2' and 'location2' are given.
           (3.2) Seasonal templates (four tables).
      (4) Eight data tables for seasonal templates with two locations.

    Visualization is handled in separate class functions.
    """

    def __init__(self, config):
        self.config = config
        self.llm = QueryLLM(config)
        self.ocr = ocr.OCR()

        # Inference settings
        self.data_path = config['inference']['data']
        self.num_samples = config['inference'].get('num_samples', -1)
        self.verbose = config['inference'].get('verbose', False)
        self.radius = config['inference'].get('radius', None)
        self.geometry = config['inference'].get('geometry', None)

    def load_samples(self):
        try:
            with open(self.data_path, 'r') as f:
                samples = json.load(f)
            return samples
        except FileNotFoundError:
            print(f"Error: The file {self.data_path} was not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode the JSON file {self.data_path}.")
        return []

    def generate_dataset(self):
        samples = self.load_samples()
        total = self.num_samples if self.num_samples > 0 else len(samples)
        for idx, sample in tqdm(enumerate(samples), total=total):
            if idx != 0:
                continue
            if self.num_samples > 0 and idx >= self.num_samples:
                break
            self.process_sample(sample, idx)

    def process_sample(self, sample, idx):
        filled = sample.get('filled_values', {})
        template = sample.get('template', '')
        question = sample.get('question', '')

        # Create a dedicated output directory for the sample.
        question_dir = os.path.join("output", f"question_{idx}")
        os.makedirs(question_dir, exist_ok=True)

        # Query the LLM to rephrase the question.
        rephrased_question = self.llm.query_llm(
            step='rephrase_question', content=question, assistant=False, verbose=False
        )
        if self.verbose:
            print(f"Processing sample {idx}: {question}")

        # --- Determine which retrieval case applies ---
        if 'season' in template or 'seasonal' in template:
            # Seasonal branch:
            if 'location2' in filled:
                # (4) Eight data tables: seasonal template with two locations.
                data_tables = self._retrieve_seasonal_eight_tables(sample)
                vis_outputs = self.visualize_seasonal(data_tables, sample, idx, eight=True)
            else:
                # (3.2) Four seasonal tables: seasonal template with a single location.
                data_tables = self._retrieve_seasonal_four_tables(sample)
                vis_outputs = self.visualize_seasonal(data_tables, sample, idx, eight=False)
        else:
            # Non-seasonal branch:
            if ('climate_variable2' not in filled and
                    'location2' not in filled and
                    'time_frame2' not in filled):
                # (1) Single data table.
                data_tables = {"table1": self._retrieve_single_data_table(sample)}
                vis_outputs = self.visualize_nonseasonal(data_tables, sample, idx)
            elif ('climate_variable2' in filled or
                  (('location2' in filled and 'time_frame2' not in filled) or
                   ('time_frame2' in filled and 'location2' not in filled))):
                # (2) Two data tables.
                data_tables = self._retrieve_two_data_tables(sample)
                vis_outputs = self.visualize_nonseasonal(data_tables, sample, idx)
            elif ('time_frame2' in filled and 'location2' in filled):
                # (3.1) Four data tables.
                data_tables = self._retrieve_four_data_tables(sample)
                vis_outputs = self.visualize_nonseasonal(data_tables, sample, idx)
            else:
                raise ValueError("Combination of filled values not implemented.")

        # Merge the individual visualization outputs.
        self.merge_visualizations(vis_outputs, template, question_dir)

        # Query the oracle for correct/incorrect answers.
        # in vis_outputs: heatmap, overlay, overlay_path, overlay_width, overlay_height, angle
        # in oracle_codes: question_dir, ppocr, llm, template, data_var1, angle1, overlay1, overlay_path1, location_description1,
        # location_description2, angle2, overlay_path2, data_var2, data_var3, data_var4, data_var5, data_var6, data_var7, data_var8
        correct_ans, incorrect_ans = oracle.oracle_codes(
            question_dir,
            self.ocr,
            self.llm,
            template,
            data_tables.get("table1")[0] if data_tables.get("table1") else None,
            vis_outputs.get("table1")[5] if vis_outputs.get("table1") else None,
            vis_outputs.get("table1")[1] if vis_outputs.get("table1") else None,
            vis_outputs.get("table1")[2] if vis_outputs.get("table1") else None,
            filled.get("location1"),
            filled.get("location2"),
            vis_outputs.get("table2")[5] if vis_outputs.get("table2") else None,
            vis_outputs.get("table2")[2] if vis_outputs.get("table2") else None,
            data_tables.get("table2")[0] if data_tables.get("table2") else None,
            data_tables.get("table3")[0] if data_tables.get("table3") else None,
            data_tables.get("table4")[0] if data_tables.get("table4") else None,
            data_tables.get("table5")[0] if data_tables.get("table5") else None,
            data_tables.get("table6")[0] if data_tables.get("table6") else None,
            data_tables.get("table7")[0] if data_tables.get("table7") else None,
            data_tables.get("table8")[0] if data_tables.get("table8") else None,
            verbose=self.verbose
        )

        # Construct the QA dictionary dynamically
        qa = {
            "question": question,
            "rephrased_question": rephrased_question,
            "filled_values": filled,
            "template": template,
            "correct_answer": correct_ans,
            "incorrect_answers": incorrect_ans,
        }

        # Dynamically add available data tables
        for i in range(1, 9):  # Loop from data_var1 to data_var8
            key = f"data_var{i}"
            if f"table{i}" in data_tables and data_tables[f"table{i}"]:
                qa[key] = data_tables[f"table{i}"][0]  # Extract the data array
                qa[f"latlong{i}"] = data_tables[f"table{i}"][1]  # Extract lat/long

        # Print QA results if verbose mode is enabled
        if self.verbose:
            utils.print_qa(qa)

    # ---------------------------
    # Data retrieval helper
    # ---------------------------
    def _retrieve_data(self, climate_var, loc_desc, time_desc):
        result = retrieve_data_from_location(
            climate_var, loc_desc, time_desc,
            self.llm, self.geometry, self.radius
        )
        if result is None:
            return None
        data, crossmodel, latlong, data_df, col_name, cell_geom = result
        data = utils.reformat_to_2d_table(data, crossmodel)
        return (data, latlong, data_df, col_name, cell_geom)

    # (1) Single Data Table
    def _retrieve_single_data_table(self, sample):
        filled = sample.get('filled_values', {})
        climate_var = filled.get('climate_variable1')
        loc_desc = filled.get('location1')
        time_desc = filled.get('time_frame1')
        return self._retrieve_data(climate_var, loc_desc, time_desc)

    # (2) Two Data Tables
    def _retrieve_two_data_tables(self, sample):
        filled = sample.get('filled_values', {})
        table1 = self._retrieve_single_data_table(sample)
        if 'climate_variable2' in filled:
            table2 = self._retrieve_data(
                filled.get('climate_variable2'),
                filled.get('location1'),
                filled.get('time_frame1')
            )
        elif 'location2' in filled and 'time_frame2' not in filled:
            table2 = self._retrieve_data(
                filled.get('climate_variable1'),
                filled.get('location2'),
                filled.get('time_frame1')
            )
        elif 'time_frame2' in filled and 'location2' not in filled:
            table2 = self._retrieve_data(
                filled.get('climate_variable1'),
                filled.get('location1'),
                filled.get('time_frame2')
            )
        else:
            table2 = None
        return {"table1": table1, "table2": table2}

    # (3.1) Four Data Tables: when both 'time_frame2' and 'location2' are provided
    def _retrieve_four_data_tables(self, sample):
        filled = sample.get('filled_values', {})
        table1 = self._retrieve_data(
            filled.get('climate_variable1'),
            filled.get('location1'),
            filled.get('time_frame1')
        )
        table2 = self._retrieve_data(
            filled.get('climate_variable1'),
            filled.get('location1'),
            filled.get('time_frame2')
        )
        table3 = self._retrieve_data(
            filled.get('climate_variable1'),
            filled.get('location2'),
            filled.get('time_frame1')
        )
        table4 = self._retrieve_data(
            filled.get('climate_variable1'),
            filled.get('location2'),
            filled.get('time_frame2')
        )
        return {"table1": table1, "table2": table2, "table3": table3, "table4": table4}

    # (3.2) Four Seasonal Data Tables: seasonal template with a single location.
    def _retrieve_seasonal_four_tables(self, sample):
        filled = sample.get('filled_values', {})
        climate_var = filled.get('climate_variable1')
        loc_desc = filled.get('location1')
        time_frame = filled.get('time_frame1')
        seasons = ['spring', 'summer', 'autumn', 'winter']
        seasonal_tables = {}
        for i, season in enumerate(seasons, start=1):
            time_desc = f"{season} in {time_frame}"
            seasonal_tables[f"table{i}"] = self._retrieve_data(climate_var, loc_desc, time_desc)
        return seasonal_tables

    # (4) Eight Data Tables: seasonal template with two locations.
    def _retrieve_seasonal_eight_tables(self, sample):
        filled = sample.get('filled_values', {})
        climate_var = filled.get('climate_variable1')
        loc_desc1 = filled.get('location1')
        loc_desc2 = filled.get('location2')
        time_frame = filled.get('time_frame1')
        seasons = ['spring', 'summer', 'autumn', 'winter']
        seasonal_tables = {}
        # For location1: tables 1-4.
        for i, season in enumerate(seasons, start=1):
            time_desc = f"{season} in {time_frame}"
            seasonal_tables[f"table{i}"] = self._retrieve_data(climate_var, loc_desc1, time_desc)
        # For location2: tables 5-8.
        for i, season in enumerate(seasons, start=5):
            time_desc = f"{season} in {time_frame}"
            seasonal_tables[f"table{i}"] = self._retrieve_data(climate_var, loc_desc2, time_desc)
        return seasonal_tables

    # ---------------------------
    # Visualization methods
    # ---------------------------
    def visualize_nonseasonal(self, data_tables, sample, idx):
        """
        Visualizes non-seasonal data tables. Depending on the number of tables (1, 2, or 4)
        different titles are assigned.
        """
        question_dir = os.path.join("output", f"question_{idx}")
        os.makedirs(question_dir, exist_ok=True)
        filled = sample.get('filled_values', {})

        # Compute a global color normalization using all available data arrays.
        valid_arrays = [tbl[0] for tbl in data_tables.values() if tbl and tbl[0] is not None]
        if not valid_arrays:
            return {}
        global_min = np.nanmin([arr.min() for arr in valid_arrays])
        global_max = np.nanmax([arr.max() for arr in valid_arrays])
        global_mid = (global_min + global_max) / 2
        color_norm = TwoSlopeNorm(vmin=global_min, vmax=global_max, vcenter=global_mid)
        vis_outputs = {}

        # Assign titles based on the number of data tables.
        if len(data_tables) == 1:
            title = filled.get('climate_variable1', "Data 1")
            tbl = data_tables["table1"]
            vis_outputs["table1"] = visualization.visualize_grids(
                question_dir, tbl[2], tbl[0], title,
                tbl[3], tbl[4],
                color_norm,
                center_lat=tbl[1][0], center_lon=tbl[1][1],
                size_km=self.radius, output_path="heatmap1", verbose=self.verbose
            )
        elif len(data_tables) == 2:
            # Titles may depend on whether a second climate variable was given.
            if 'climate_variable2' in filled:
                title1 = filled.get('climate_variable1')
                title2 = filled.get('climate_variable2')
            else:
                if 'time_frame2' in filled:
                    title1 = filled.get('time_frame1')
                    title2 = filled.get('time_frame2')
                else:
                    title1 = filled.get('location1')
                    title2 = filled.get('location2')
            for key, title in zip(["table1", "table2"], [title1, title2]):
                tbl = data_tables[key]
                vis_outputs[key] = visualization.visualize_grids(
                    question_dir, tbl[2], tbl[0], title,
                    tbl[3], tbl[4],
                    color_norm,
                    center_lat=tbl[1][0], center_lon=tbl[1][1],
                    size_km=self.radius, output_path=f"heatmap_{key}", verbose=self.verbose
                )
        elif len(data_tables) == 4:
            title1 = f"{filled.get('location1')} in {filled.get('time_frame1')}"
            title2 = f"{filled.get('location1')} in {filled.get('time_frame2')}"
            title3 = f"{filled.get('location2')} in {filled.get('time_frame1')}"
            title4 = f"{filled.get('location2')} in {filled.get('time_frame2')}"
            titles = {"table1": title1, "table2": title2, "table3": title3, "table4": title4}
            for key in ["table1", "table2", "table3", "table4"]:
                tbl = data_tables[key]
                vis_outputs[key] = visualization.visualize_grids(
                    question_dir, tbl[2], tbl[0], titles[key],
                    tbl[3], tbl[4],
                    color_norm,
                    center_lat=tbl[1][0], center_lon=tbl[1][1],
                    size_km=self.radius, output_path=f"heatmap_{key}", verbose=self.verbose
                )
        return vis_outputs

    def visualize_seasonal(self, data_tables, sample, idx, eight=False):
        """
        Visualizes seasonal data tables. If eight tables are provided then the first four are for
        location1 and the next four for location2.
        """
        question_dir = os.path.join("output", f"question_{idx}")
        os.makedirs(question_dir, exist_ok=True)
        filled = sample.get('filled_values', {})
        time_frame = filled.get('time_frame1')
        valid_arrays = [tbl[0] for tbl in data_tables.values() if tbl and tbl[0] is not None]
        if not valid_arrays:
            return {}
        global_min = np.nanmin([arr.min() for arr in valid_arrays])
        global_max = np.nanmax([arr.max() for arr in valid_arrays])
        global_mid = (global_min + global_max) / 2
        color_norm = TwoSlopeNorm(vmin=global_min, vmax=global_max, vcenter=global_mid)
        vis_outputs = {}
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        if not eight:
            for i, season in enumerate(seasons, start=1):
                title = f"{season} in {filled.get('location1', time_frame)}"
                tbl = data_tables.get(f"table{i}")
                if tbl is not None:
                    vis_outputs[f"table{i}"] = visualization.visualize_grids(
                        question_dir, tbl[2], tbl[0], title,
                        tbl[3], tbl[4],
                        color_norm,
                        center_lat=tbl[1][0], center_lon=tbl[1][1],
                        size_km=self.radius, output_path=f"heatmap_table{i}", verbose=self.verbose
                    )
        else:
            for i, season in enumerate(seasons, start=1):
                title = f"{season} in {filled.get('location1')}"
                tbl = data_tables.get(f"table{i}")
                if tbl is not None:
                    vis_outputs[f"table{i}"] = visualization.visualize_grids(
                        question_dir, tbl[2], tbl[0], title,
                        tbl[3], tbl[4],
                        color_norm,
                        center_lat=tbl[1][0], center_lon=tbl[1][1],
                        size_km=self.radius, output_path=f"heatmap_table{i}", verbose=self.verbose
                    )
            for i, season in enumerate(seasons, start=5):
                title = f"{season} in {filled.get('location2')}"
                tbl = data_tables.get(f"table{i}")
                if tbl is not None:
                    vis_outputs[f"table{i}"] = visualization.visualize_grids(
                        question_dir, tbl[2], tbl[0], title,
                        tbl[3], tbl[4],
                        color_norm,
                        center_lat=tbl[1][0], center_lon=tbl[1][1],
                        size_km=self.radius, output_path=f"heatmap_table{i}", verbose=self.verbose
                    )
        return vis_outputs

    def merge_visualizations(self, vis_outputs, template, question_dir):
        """
        Merge individual visualization images (heatmaps and overlays) into composite images.
        """

        def get_image(viz_result, index):
            return viz_result[index] if viz_result is not None else None

        merged_heatmap = None
        merged_overlay = None

        if 'table3' in vis_outputs and get_image(vis_outputs.get('table3'), 0) is not None and get_image(vis_outputs.get('table4'), 0) is not None:
            hm1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 0),
                                          get_image(vis_outputs.get('table2'), 0))
            hm2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 0),
                                          get_image(vis_outputs.get('table4'), 0))
            vertical = not ('season' in template or 'seasonal' in template)
            merged_heatmap = utils.merge_two_figures(hm1, hm2, vertical=vertical)

            ov1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 1),
                                          get_image(vis_outputs.get('table2'), 1))
            ov2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 1),
                                          get_image(vis_outputs.get('table4'), 1))
            merged_overlay = utils.merge_two_figures(ov1, ov2, vertical=vertical)
        elif 'table2' in vis_outputs:
            merged_heatmap = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 0),
                                                     get_image(vis_outputs.get('table2'), 0))
            merged_overlay = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 1),
                                                     get_image(vis_outputs.get('table2'), 1))
        if merged_heatmap is not None:
            merged_heatmap.save(os.path.join(question_dir, 'heatmap_merged.png'))
        if merged_overlay is not None:
            merged_overlay.save(os.path.join(question_dir, 'heatmap_overlay_merged.png'))
        if self.verbose:
            print("Merged visualizations saved.")


# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    print("Python", sys.version, "Torch", torch.__version__)

    # Load configuration from YAML.
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print("Error reading config.yaml:", e)
        sys.exit(1)

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Benchmark Dataset Generator")
    parser.add_argument('--llm', type=str, default="gpt-4o-mini", help="Set LLM model")
    parser.add_argument('--n', type=int, default=-1, help="Number of samples to generate")
    parser.add_argument('--data', type=str, default="test_filled_questions.json", help="Data file")
    parser.add_argument('--verbose', dest='verbose', action='store_true', help="Enable verbose output")
    cmd_args = parser.parse_args()
    cmd_args.data = os.path.join("data", cmd_args.data)

    # Override config values with command-line arguments.
    config['models']['llm'] = cmd_args.llm or config['models'].get('llm')
    config['inference']['num_samples'] = cmd_args.n if cmd_args.n is not None else config['inference'].get('n')
    config['inference']['data'] = cmd_args.data or config['inference'].get('data')
    config['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else config['inference'].get('verbose', False)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = torch.cuda.device_count()
    assert world_size == 1, "Only one GPU is supported."
    print("Device:", device)
    print("Using", world_size, "GPU(s)")

    generator = BenchmarkDatasetGenerator(config)
    generator.generate_dataset()





# import numpy as np
# import torch
# import json
#
# import os
# os.environ['USE_PYGEOS'] = '0'
# import geopandas as gpd
#
# from tqdm import tqdm
# from matplotlib.colors import TwoSlopeNorm
# import yaml
# import json
# import argparse
# import sys
#
# from data_retrieval import retrieve_data_from_location
# from query_llm import QueryLLM
# import utils
# import visualization
# import ocr
# import oracle
#
#
# def dataloader(json_file_path):
#     try:
#         with open(json_file_path, 'r') as file:
#             data = json.load(file)
#             for entry in data:
#                 yield entry
#     except FileNotFoundError:
#         print(f"Error: The file {json_file_path} was not found.")
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode the JSON file {json_file_path}.")
#
#
# def generate_dataset(args):
#     llm = QueryLLM(args)
#     ppocr = ocr.OCR()
#
#     for i, data_sample in tqdm(enumerate(dataloader(args['inference']['data']))):
#         if i != 7:
#             continue
#
#         question_dir = f'output/question_{i}'
#         os.makedirs(question_dir, exist_ok=True)
#
#         question = data_sample['question']
#         filled_values = data_sample['filled_values']
#         template = data_sample['template']
#         rephrased_question = llm.query_llm(step='rephrase_question', content=question, assistant=False, verbose=False)
#         if args['inference']['verbose']:
#             print(f'{utils.Colors.HEADER}Processing data sample {i}. Question {question}{utils.Colors.ENDC}')
#
#         # Load data for climate_variable1
#         climate_variable1 = filled_values['climate_variable1']
#         location_description1 = filled_values['location1']
#         time_period1 = filled_values['time_frame1']
#
#         data_var2, data_var3, data_var4, data_var5, data_var6, data_var7, data_var8 = None, None, None, None, None, None, None
#         data_df2, data_df3, data_df4, data_df5, data_df6, data_df7, data_df8 = None, None, None, None, None, None, None
#         heatmap2, overlay2, overlay_path2, data_df2, df_col_name2, cell_geometries2, angle2, heatmap3, heatmap5 = None, None, None, None, None, None, None, None, None
#
#         if 'season' in template or 'seasonal' in template:
#             # Initialize data variables for 4 seasons
#             data_vars = {f"data_var{i}": None for i in range(1, 9)}
#             data_dfs = {f"data_df{i}": None for i in range(1, 9)}
#             crossmodel_indices = {f"crossmodel_indices{i}": None for i in range(1, 9)}
#             df_col_names = {f"df_col_name{i}": None for i in range(1, 9)}
#             location_description2 = None
#
#             for i, season in enumerate(['spring', 'summer', 'autumn', 'winter', 'spring', 'summer', 'autumn', 'winter'], start=1):
#                 if i > 3 and 'location2' not in filled_values:
#                     break
#                 key_var = f"data_var{i}"
#                 key_df = f"data_df{i}"
#                 key_crossmodel_indices = f"crossmodel_indices{i}"
#                 key_df_col_name = f"df_col_name{i}"
#
#                 # Retrieve data for each season. We name some of them as _1 consistently because they refer to the same location
#                 if i < 3:
#                     data_vars[key_var], crossmodel_indices[key_crossmodel_indices], latlong1, data_dfs[key_df], df_col_names[key_df_col_name], cell_geometries1 = (
#                         retrieve_data_from_location(climate_variable1, location_description1, f"{season} in {time_period1}", llm, args['inference']['geometry'], args['inference']['radius']))
#                 else: # Retrieve data for the second location
#                     location_description2 = filled_values['location2']
#                     data_vars[key_var], crossmodel_indices[key_crossmodel_indices], latlong2, data_dfs[key_df], df_col_names[key_df_col_name], cell_geometries2 = (
#                         retrieve_data_from_location(climate_variable1, location_description2, f"{season} in {time_period1}", llm, args['inference']['geometry'], args['inference']['radius']))
#                 data_vars[key_var] = utils.reformat_to_2d_table(data_vars[key_var], crossmodel_indices[key_crossmodel_indices])
#
#             data_var1, data_var2, data_var3, data_var4 = data_vars['data_var1'], data_vars['data_var2'], data_vars['data_var3'], data_vars['data_var4']
#             data_df1, data_df2, data_df3, data_df4 = data_dfs['data_df1'], data_dfs['data_df2'], data_dfs['data_df3'], data_dfs['data_df4']
#             df_col_name1, df_col_name2, df_col_name3, df_col_name4 = df_col_names['df_col_name1'], df_col_names['df_col_name2'], df_col_names['df_col_name3'], df_col_names['df_col_name4']
#             if location_description2:
#                 data_var5, data_var6, data_var7, data_var8 = data_vars['data_var5'], data_vars['data_var6'], data_vars['data_var7'], data_vars['data_var8']
#                 data_df5, data_df6, data_df7, data_df8 = data_dfs['data_df5'], data_dfs['data_df6'], data_dfs['data_df7'], data_dfs['data_df8']
#                 df_col_name5, df_col_name6, df_col_name7, df_col_name8 = df_col_names['df_col_name5'], df_col_names['df_col_name6'], df_col_names['df_col_name7'], df_col_names['df_col_name8']
#         else:
#             data_var1, crossmodel_indices1, latlong1, data_df1, df_col_name1, cell_geometries1 = retrieve_data_from_location(climate_variable1, location_description1, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
#             data_var1 = utils.reformat_to_2d_table(data_var1, crossmodel_indices1)
#
#             # Load data for climate_variable2
#             location_description2 = None
#             if 'climate_variable2' in filled_values:
#                 climate_variable2 = filled_values['climate_variable2']
#                 if time_period1 not in utils.full_time_frames[climate_variable2]:
#                     continue
#                 data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable2, location_description1, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
#                 data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
#             else:
#                 if 'location2' in filled_values and 'time_frame2' not in filled_values:
#                     # Same climate variable but at two different locations
#                     assert 'time_frame2' not in filled_values
#                     location_description2 = filled_values['location2']
#                     data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable1, location_description2, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
#                     data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
#                 if 'time_frame2' in filled_values and 'location2' not in filled_values:
#                     # Same climate variable but at two different time periods
#                     assert 'location2' not in filled_values
#                     time_period2 = filled_values['time_frame2']
#                     data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable1, location_description1, time_period2, llm, args['inference']['geometry'], args['inference']['radius'])
#                     data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
#                 if 'time_frame2' in filled_values and 'location2' in filled_values:
#                     location_description2 = filled_values['location2']
#                     time_period2 = filled_values['time_frame2']
#                     data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable1, location_description1, time_period2, llm, args['inference']['geometry'], args['inference']['radius'])
#                     data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
#                     data_var3, crossmodel_indices3, latlong3, data_df3, df_col_name3, cell_geometries3 = retrieve_data_from_location(climate_variable1, location_description2, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
#                     data_var3 = utils.reformat_to_2d_table(data_var3, crossmodel_indices3)
#                     data_var4, crossmodel_indices4, latlong4, data_df4, df_col_name4, cell_geometries4 = retrieve_data_from_location(climate_variable1, location_description2, time_period2, llm, args['inference']['geometry'], args['inference']['radius'])
#                     data_var4 = utils.reformat_to_2d_table(data_var4, crossmodel_indices4)
#
#         qa = {"question": question, "rephrased_question": rephrased_question, "filled_values": filled_values, "template": template, "data_var1": data_var1[::-1], "latlong1": latlong1}
#         if data_var2 is not None:
#             qa["data_var2"] = data_var2[::-1]
#         if data_var3 is not None:
#             qa["data_var3"] = data_var3[::-1]
#         if data_var4 is not None:
#             qa["data_var4"] = data_var4[::-1]
#
#         """
#         The following answers come from one of the top place names shown on the actual map
#         """
#         if 'season' in template or 'seasonal' in template:
#             if 'location2' in filled_values:
#                 title1, title2, title3, title4 = 'Spring in ' + filled_values['location1'], 'Summer in ' + filled_values['location1'], 'Autumn in ' + filled_values['location1'], 'Winter in ' + filled_values['location1']
#                 title5, title6, title7, title8 = 'Spring in ' + filled_values['location2'], 'Summer in ' + filled_values['location2'], 'Autumn in ' + filled_values['location2'], 'Winter in ' + filled_values['location2']
#             else:
#                 title1, title2, title3, title4 = 'Spring in ' + filled_values['time_frame1'], 'Summer in ' + filled_values['time_frame1'], 'Autumn in ' + filled_values['time_frame1'], 'Winter in ' + filled_values['time_frame1']
#         elif 'location2' in filled_values and 'time_frame2' not in filled_values:
#             title1, title2 = filled_values['location1'], filled_values['location2']
#         elif 'time_frame2' in filled_values and 'location2' not in filled_values:
#             title1, title2 = filled_values['time_frame1'], filled_values['time_frame2']
#         elif 'location2' in filled_values and 'time_frame2' in filled_values:
#             title1, title2, title3, title4 = filled_values['location1'] + ' in ' + filled_values['time_frame1'], filled_values['location1'] + ' in ' + filled_values['time_frame2'], filled_values['location2'] + ' in ' + filled_values['time_frame1'], filled_values['location2'] + ' in ' + filled_values['time_frame2']
#         else:
#             title1, title2 = filled_values['climate_variable1'], filled_values['climate_variable2'] if 'climate_variable2' in filled_values else None
#
#         # Define a global minimum, maximum, and midpoint across both datasets
#         valid_data_vars = [var for var in [data_var1, data_var2, data_var3, data_var4, data_var5, data_var6, data_var7, data_var8] if var is not None]
#         global_min = np.nanmin([var.min().min() for var in valid_data_vars])
#         global_max = np.nanmax([var.max().max() for var in valid_data_vars])
#         global_midpoint = (global_max + global_min) / 2
#         color_norm = TwoSlopeNorm(vmin=global_min, vmax=global_max, vcenter=global_midpoint)
#
#         heatmap1, overlay1, overlay_path1, overlay_width1, overlay_height1, angle1 = visualization.visualize_grids(question_dir, data_df1, data_var1, title1, df_col_name1, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap1', verbose=args['inference']['verbose'])
#         if ('climate_variable2' in filled_values) or ('location2' in filled_values and 'time_frame2' not in filled_values) or ('time_frame2' in filled_values and 'location2' not in filled_values):
#             heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2, angle2 = visualization.visualize_grids(question_dir, data_df2, data_var2, title2, df_col_name2, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])
#
#         if ('season' in template or 'seasonal' in template) or ('location2' in filled_values and 'time_frame2' in filled_values):
#             if 'location2' in filled_values and 'time_frame2' in filled_values:
#                 print('!!!!!!!!')
#                 heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2, angle2 = visualization.visualize_grids(question_dir, data_df2, data_var2, title2, df_col_name2, cell_geometries1, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])
#                 heatmap3, overlay3, overlay_path3, overlay_width3, overlay_height3, angle3 = visualization.visualize_grids(question_dir, data_df3, data_var3, title3, df_col_name3, cell_geometries2, color_norm, center_lat=latlong3[0], center_lon=latlong3[1], size_km=args['inference']['radius'], output_path='heatmap3', verbose=args['inference']['verbose'])
#                 heatmap4, overlay4, overlay_path4, overlay_width4, overlay_height4, angle4 = visualization.visualize_grids(question_dir, data_df4, data_var4, title4, df_col_name4, cell_geometries2, color_norm, center_lat=latlong4[0], center_lon=latlong4[1], size_km=args['inference']['radius'], output_path='heatmap4', verbose=args['inference']['verbose'])
#             else:
#                 heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2, angle2 = visualization.visualize_grids(question_dir, data_df2, data_var2, title2, df_col_name2, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])
#                 heatmap3, overlay3, overlay_path3, overlay_width3, overlay_height3, angle3 = visualization.visualize_grids(question_dir, data_df3, data_var3, title3, df_col_name3, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap3', verbose=args['inference']['verbose'])
#                 heatmap4, overlay4, overlay_path4, overlay_width4, overlay_height4, angle4 = visualization.visualize_grids(question_dir, data_df4, data_var4, title4, df_col_name4, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap4', verbose=args['inference']['verbose'])
#
#             if data_df5 is not None: # Seasons of the second location
#                 heatmap5, overlay5, overlay_path5, overlay_width5, overlay_height5, angle5 = visualization.visualize_grids(question_dir, data_df5, data_var5, title5, df_col_name5, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap5', verbose=args['inference']['verbose'])
#                 heatmap6, overlay6, overlay_path6, overlay_width6, overlay_height6, angle6 = visualization.visualize_grids(question_dir, data_df6, data_var6, title6, df_col_name6, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap6', verbose=args['inference']['verbose'])
#                 heatmap7, overlay7, overlay_path7, overlay_width7, overlay_height7, angle7 = visualization.visualize_grids(question_dir, data_df7, data_var7, title7, df_col_name7, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap7', verbose=args['inference']['verbose'])
#                 heatmap8, overlay8, overlay_path8, overlay_width8, overlay_height8, angle8 = visualization.visualize_grids(question_dir, data_df8, data_var8, title8, df_col_name8, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap8', verbose=args['inference']['verbose'])
#
#         if heatmap3 is not None and heatmap4 is not None:
#             heatmap_merged = utils.merge_two_figures(
#                 utils.merge_two_figures(heatmap1, heatmap2), utils.merge_two_figures(heatmap3, heatmap4), vertical=not ('season' in template or 'seasonal' in template))
#             overlay_merged = utils.merge_two_figures(
#                 utils.merge_two_figures(overlay1, overlay2), utils.merge_two_figures(overlay3, overlay4), vertical=not ('season' in template or 'seasonal' in template))
#
#             if heatmap5 is not None:
#                 heatmap_merged2 = utils.merge_two_figures(
#                     utils.merge_two_figures(heatmap5, heatmap6), utils.merge_two_figures(heatmap7, heatmap8), vertical=False)
#                 overlay_merged2 = utils.merge_two_figures(
#                     utils.merge_two_figures(overlay5, overlay6), utils.merge_two_figures(overlay7, overlay8), vertical=False)
#                 heatmap_merged = utils.merge_two_figures(heatmap_merged, heatmap_merged2, vertical=True)
#                 overlay_merged = utils.merge_two_figures(overlay_merged, overlay_merged2, vertical=True)
#
#             heatmap_merged.save(os.path.join(question_dir, 'heatmap_merged.png'))
#             overlay_merged.save(os.path.join(question_dir, 'heatmap_overlay_merged.png'))
#         elif heatmap2 is not None:
#             heatmap_merged = utils.merge_two_figures(heatmap1, heatmap2)
#             overlay_merged = utils.merge_two_figures(overlay1, overlay2)
#             heatmap_merged.save(os.path.join(question_dir, 'heatmap_merged.png'))
#             overlay_merged.save(os.path.join(question_dir, 'heatmap_overlay_merged.png'))
#         if args['inference']['verbose']:
#             print("Merged heatmap and overlay saved.")
#
#         """
#         The following answers come from one of the following relative locations: upper-left, upper-mid, upper-right, mid-left, center, mid-right, lower-left, lower-mid, lower-right
#         """
#         correct_answer, incorrect_answers = oracle.oracle_codes(question_dir, ppocr, llm, template, data_var1, angle1, overlay1, overlay_path1, location_description1, location_description2,
#                                                                 angle2, overlay_path2, data_var2, data_var3, data_var4, data_var5, data_var6, data_var7, data_var8, verbose=args['inference']['verbose'])
#         qa["correct_answer"] = correct_answer
#         qa["incorrect_answers"] = incorrect_answers
#
#         if args['inference']['verbose']:
#             utils.print_qa(qa)
#
#
# if __name__ == "__main__":
#     print("Python", sys.version, 'Torch', torch.__version__)
#     # Load hyperparameters
#     try:
#         with open('config.yaml', 'r') as file:
#             args = yaml.safe_load(file)
#     except Exception as e:
#         print('Error reading the config file')
#
#     # Command-line argument parsing
#     parser = argparse.ArgumentParser(description='Command line arguments')
#     parser.add_argument('--llm', type=str, default="gpt-4o-mini", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
#     parser.add_argument('--n', type=int, default=-1, help='Set number of samples to generate. Default is -1, which generates all samples')
#     parser.add_argument('--data', type=str, default="test_filled_questions.json", help='Set data file to generate samples from')
#     parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
#     cmd_args = parser.parse_args()
#     cmd_args.data = f'./data/{cmd_args.data}'
#
#     # Override args from config.yaml with command-line arguments if provided
#     args['models']['llm'] = cmd_args.llm if cmd_args.llm is not None else args['models']['llm']
#     args['inference']['num_samples'] = cmd_args.n if cmd_args.n is not None else args['inference']['n']
#     args['inference']['data'] = cmd_args.data if cmd_args.data is not None else args['inference']['data']
#     args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
#
#     torch.manual_seed(0)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     world_size = torch.cuda.device_count()
#     assert world_size == 1
#     print('device', device)
#     print('torch.distributed.is_available', torch.distributed.is_available())
#     print('Using %d GPUs' % (torch.cuda.device_count()))
#
#     # Start inference
#     print(args)
#     generate_dataset(args)