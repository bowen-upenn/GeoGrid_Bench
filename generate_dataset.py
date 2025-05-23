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

            if isinstance(samples, dict):
                all_samples = []
                for key, curr_list in samples.items():
                    for item in curr_list:
                        item['template'] = key
                    all_samples.extend(curr_list)
                samples = all_samples

            return samples
        except FileNotFoundError:
            print(f"Error: The file {self.data_path} was not found.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode the JSON file {self.data_path}.")
        return []

    def generate_dataset(self):
        samples = self.load_samples()
        total = self.num_samples if self.num_samples > 0 else len(samples)
        all_failed_samples = []
        for idx, sample in tqdm(enumerate(samples), total=total):
            if idx != 0:
                continue
            if self.num_samples > 0 and idx >= self.num_samples:
                break
            try:
                self.process_sample(sample, idx)
            except Exception as e:
                all_failed_samples.append(f'{utils.Colors.OKBLUE}Error Processing {idx}{utils.Colors.ENDC}{e}')
                print(f'{utils.Colors.OKBLUE}Error Processing {idx}{utils.Colors.ENDC}{e}')

        if all_failed_samples:
            print(f'{utils.Colors.FAIL}Failed Samples:{utils.Colors.ENDC}')
            for failed in all_failed_samples:
                print(failed)

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
                vis_outputs, title = self.visualize_seasonal(data_tables, sample, idx, eight=True)
            else:
                # (3.2) Four seasonal tables: seasonal template with a single location.
                data_tables = self._retrieve_seasonal_four_tables(sample)
                vis_outputs, title = self.visualize_seasonal(data_tables, sample, idx, eight=False)
        else:
            # Non-seasonal branch:
            if ('climate_variable2' not in filled and
                    'location2' not in filled and
                    'time_frame2' not in filled):
                # (1) Single data table.
                data_tables = {"table1": self._retrieve_single_data_table(sample)}
                vis_outputs, title = self.visualize_nonseasonal(data_tables, sample, idx)
            elif ('climate_variable2' in filled or
                  (('location2' in filled and 'time_frame2' not in filled) or
                   ('time_frame2' in filled and 'location2' not in filled))):
                # (2) Two data tables.
                data_tables = self._retrieve_two_data_tables(sample)
                vis_outputs, title = self.visualize_nonseasonal(data_tables, sample, idx)
            elif ('time_frame2' in filled and 'location2' in filled):
                # (3.1) Four data tables.
                data_tables = self._retrieve_four_data_tables(sample)
                vis_outputs, title = self.visualize_nonseasonal(data_tables, sample, idx)
            else:
                raise ValueError("Combination of filled values not implemented.")

        # Merge the individual visualization outputs.
        self.merge_visualizations(vis_outputs, template, question_dir)

        # Query the oracle for correct/incorrect answers.
        # in vis_outputs: heatmap, heatmap_with_text, overlay, overlay_path, overlay_width, overlay_height, angle, heatmap_rect
        # in oracle_codes: question_dir, ppocr, llm, template, data_var1, angle1, heatmap_rect1, overlay1, overlay_path1, location_description1,
        # location_description2, angle2, overlay_path2, data_var2, data_var3, data_var4, data_var5, data_var6, data_var7, data_var8,
        print('data_tables', data_tables)
        correct_ans, incorrect_ans = oracle.oracle_codes(
            question_dir,
            self.ocr,
            self.llm,
            template,
            data_tables.get("table1")[0] if data_tables.get("table1") else None,
            vis_outputs.get("table1")[6] if vis_outputs.get("table1") else None,
            vis_outputs.get("table1")[7] if vis_outputs.get("table1") else None,
            vis_outputs.get("table1")[2] if vis_outputs.get("table1") else None,
            vis_outputs.get("table1")[3] if vis_outputs.get("table1") else None,
            filled.get("location1"),
            filled.get("location2"),
            vis_outputs.get("table2")[6] if vis_outputs.get("table2") else None,
            vis_outputs.get("table2")[3] if vis_outputs.get("table2") else None,
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
            "question_dir": question_dir,
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
        utils.print_and_save_qa(qa, title, self.radius, verbose=self.verbose)

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
        global_min = np.nanmin([arr.min().min() for arr in valid_arrays])
        global_max = np.nanmax([arr.max().max() for arr in valid_arrays])
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
            title = [title1, title2]
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
            title = [title1, title2, title3, title4]
        return vis_outputs, title

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
        global_min = np.nanmin([arr.min().min() for arr in valid_arrays])
        global_max = np.nanmax([arr.max().max() for arr in valid_arrays])
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
        return vis_outputs, title

    def merge_visualizations(self, vis_outputs, template, question_dir):
        """
        Merge individual visualization images (heatmaps and overlays) into composite images.
        """

        def get_image(viz_result, index):
            return viz_result[index] if viz_result is not None else None

        merged_heatmap = None
        merged_heatmap_with_text = None
        merged_overlay = None

        if 'table8' in vis_outputs:
            # Merge heatmaps for eight tables.
            hm1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 0),
                                          get_image(vis_outputs.get('table2'), 0))
            hm2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 0),
                                          get_image(vis_outputs.get('table4'), 0))
            hm3 = utils.merge_two_figures(get_image(vis_outputs.get('table5'), 0),
                                          get_image(vis_outputs.get('table6'), 0))
            hm4 = utils.merge_two_figures(get_image(vis_outputs.get('table7'), 0),
                                          get_image(vis_outputs.get('table8'), 0))
            merged_heatmap1 = utils.merge_two_figures(hm1, hm2)
            merged_heatmap2 = utils.merge_two_figures(hm3, hm4)
            merged_heatmap = utils.merge_two_figures(merged_heatmap1, merged_heatmap2, vertical=True)

            hmt1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 1),
                                           get_image(vis_outputs.get('table2'), 1))
            hmt2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 1),
                                           get_image(vis_outputs.get('table4'), 1))
            hmt3 = utils.merge_two_figures(get_image(vis_outputs.get('table5'), 1),
                                           get_image(vis_outputs.get('table6'), 1))
            hmt4 = utils.merge_two_figures(get_image(vis_outputs.get('table7'), 1),
                                           get_image(vis_outputs.get('table8'), 1))
            merged_heatmap_with_text1 = utils.merge_two_figures(hmt1, hmt2)
            merged_heatmap_with_text2 = utils.merge_two_figures(hmt3, hmt4)
            merged_heatmap_with_text = utils.merge_two_figures(merged_heatmap_with_text1, merged_heatmap_with_text2, vertical=True)

            ov1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 2),
                                          get_image(vis_outputs.get('table2'), 2))
            ov2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 2),
                                          get_image(vis_outputs.get('table4'), 2))
            ov3 = utils.merge_two_figures(get_image(vis_outputs.get('table5'), 2),
                                          get_image(vis_outputs.get('table6'), 2))
            ov4 = utils.merge_two_figures(get_image(vis_outputs.get('table7'), 2),
                                          get_image(vis_outputs.get('table8'), 2))
            merged_overlay1 = utils.merge_two_figures(ov1, ov2)
            merged_overlay2 = utils.merge_two_figures(ov3, ov4)
            merged_overlay = utils.merge_two_figures(merged_overlay1, merged_overlay2, vertical=True)

        elif 'table3' in vis_outputs and get_image(vis_outputs.get('table3'), 0) is not None and get_image(vis_outputs.get('table4'), 0) is not None:
            hm1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 0),
                                          get_image(vis_outputs.get('table2'), 0))
            hm2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 0),
                                          get_image(vis_outputs.get('table4'), 0))
            vertical = not ('season' in template or 'seasonal' in template)
            merged_heatmap = utils.merge_two_figures(hm1, hm2, vertical=vertical)

            hmt1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 1),
                                           get_image(vis_outputs.get('table2'), 1))
            hmt2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 1),
                                           get_image(vis_outputs.get('table4'), 1))
            vertical = not ('season' in template or 'seasonal' in template)
            merged_heatmap_with_text = utils.merge_two_figures(hmt1, hmt2, vertical=vertical)

            ov1 = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 2),
                                          get_image(vis_outputs.get('table2'), 2))
            ov2 = utils.merge_two_figures(get_image(vis_outputs.get('table3'), 2),
                                          get_image(vis_outputs.get('table4'), 2))
            merged_overlay = utils.merge_two_figures(ov1, ov2, vertical=vertical)

        elif 'table2' in vis_outputs:
            merged_heatmap = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 0),
                                                     get_image(vis_outputs.get('table2'), 0))
            merged_heatmap_with_text = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 1),
                                                               get_image(vis_outputs.get('table2'), 1))
            merged_overlay = utils.merge_two_figures(get_image(vis_outputs.get('table1'), 2),
                                                     get_image(vis_outputs.get('table2'), 2))

        if merged_heatmap is not None:
            merged_heatmap.save(os.path.join(question_dir, f'heatmap_merged_radius{int(self.radius)}.png'))
        if merged_heatmap_with_text is not None:
            merged_heatmap_with_text.save(os.path.join(question_dir, f'heatmap_with_text_merged_radius{int(self.radius)}.png'))
        if merged_overlay is not None:
            merged_overlay.save(os.path.join(question_dir, f'heatmap_overlay_merged_radius{int(self.radius)}.png'))
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
    parser.add_argument('--output', type=str, default="data/", help="Output path")
    parser.add_argument('--radius', type=float, help="Radius for data retrieval")
    parser.add_argument('--verbose', dest='verbose', action='store_true', help="Enable verbose output")
    cmd_args = parser.parse_args()
    cmd_args.data = os.path.join("data", cmd_args.data)

    # Override config values with command-line arguments.
    config['models']['llm'] = cmd_args.llm or config['models'].get('llm')
    config['inference']['num_samples'] = cmd_args.n if cmd_args.n is not None else config['inference'].get('n')
    config['inference']['data'] = cmd_args.data or config['inference'].get('data')
    config['inference']['output'] = cmd_args.output or config['inference'].get('output')
    config['inference']['radius'] = cmd_args.radius if cmd_args.radius is not None else config['inference'].get('radius')
    config['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else config['inference'].get('verbose', False)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = torch.cuda.device_count()
    assert world_size == 1

    generator = BenchmarkDatasetGenerator(config)
    generator.generate_dataset()
