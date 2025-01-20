from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import random
import re
import ast
import math

import utils


def find_row_col_boxes(ocr_results):
    # Extract row and column names
    row_col_indices = []
    for entry in ocr_results:
        box, (text, confidence) = entry
        if re.match(r'^(R\d{1,}|C\d{1,})', text.strip()):  # Adjust regex for matching
            row_col_indices.append(entry)
    return row_col_indices


def find_heatmap_region(ocr_results):
    # Extract bounding box coordinates for rows (Rxxx) and columns (Cxxx)
    row_col_indices = find_row_col_boxes(ocr_results)
    row_boxes = [entry[0] for entry in row_col_indices if entry[1][0].startswith('R')]
    col_boxes = [entry[0] for entry in row_col_indices if entry[1][0].startswith('C')]

    uppermost_row = min(box[0][1] for box in row_boxes)
    lowermost_row = max(box[-1][1] for box in row_boxes)

    leftmost_col = min(box[0][0] for box in col_boxes)
    rightmost_col = max(box[1][0] for box in col_boxes)

    # # Construct the rectangle
    # heatmap_rect = [
    #     [leftmost_col, uppermost_row],  # Top-left
    #     [rightmost_col, uppermost_row],  # Top-right
    #     [rightmost_col, lowermost_row],  # Bottom-right
    #     [leftmost_col, lowermost_row],  # Bottom-left
    # ]
    # Step 1: Top-left remains the same
    top_left = [leftmost_col, uppermost_row]

    # Step 2: Calculate the top-right
    angle_10_deg = math.radians(10)
    top_right = [
        rightmost_col,
        uppermost_row - (rightmost_col - leftmost_col) * math.tan(angle_10_deg)
    ]

    # Step 3: Calculate the lower-left
    angle_80_deg = math.radians(80)
    lower_left = [
        leftmost_col + (lowermost_row - uppermost_row) / math.tan(angle_80_deg),
        lowermost_row
    ]

    # Step 4: Calculate the lower-right
    lower_right = [
        rightmost_col + (lowermost_row - uppermost_row) / math.tan(angle_80_deg),
        lowermost_row - (rightmost_col - leftmost_col) * math.tan(angle_10_deg)
    ]

    # Construct the rectangle
    heatmap_rect = [top_left, top_right, lower_right, lower_left]
    print('heatmap_rect', heatmap_rect)
    return heatmap_rect


def is_point_in_box(point, vertices):
    """
        Check if a point (x, y) is inside a convex quadrilateral.

        Args:
        - point: Tuple (x, y) of the point to check.
        - vertices: List of four vertices (in order) defining the quadrilateral.

        Returns:
        - True if the point is inside, False otherwise.
        """

    # x_min = min(box[0][0], box[3][0])
    # x_max = max(box[1][0], box[2][0])
    # y_min = min(box[0][1], box[1][1])
    # y_max = max(box[2][1], box[3][1])
    # return x_min <= x <= x_max and y_min <= y <= y_max

    def cross_product(p1, p2, p3):
        """Calculate the cross product of vector (p2 - p1) and (p3 - p1)."""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    x, y = point
    signs = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]  # Next vertex (wrap around to the start)
        signs.append(cross_product(p1, p2, (x, y)) > 0)

    # Check if all cross products have the same sign
    return all(signs) or not any(signs)


def is_within_heatmap_rect(bbox, heatmap_rect):
    """Check if a bounding box is fully within the heatmap rectangle.
    boxes should have the format of [upper-left, upper-right, lower-right, lower-left] each of which is [col, row]
    """
    for corner in bbox:
        if is_point_in_box(corner, heatmap_rect):
            return True
    return False


def divide_heatmap_into_regions(heatmap_rect, ocr_results):
    """
    Divide the heatmap_rect into 9 regions and assign OCR names to each region.

    :param heatmap_rect: The bounding box of the heatmap in the format [x_min, y_min, x_max, y_max]
    :param ocr_results: OCR results with bounding boxes and detected text
    :return: Dictionary of 9 regions with OCR names assigned to them
    """
    top_left, top_right, lower_right, lower_left = heatmap_rect

    height = lower_left[1] - top_left[1]
    width = top_right[0] - top_left[0]
    row_third = height / 3
    col_third = width / 3

    # Calculate the boundaries for the 9 regions, considering the slant
    def interpolate_point(p1, p2, factor):
        """Interpolate between two points by a given factor."""
        return [p1[0] + (p2[0] - p1[0]) * factor, p1[1] + (p2[1] - p1[1]) * factor]

    region_boundaries = {
        "upper-left": [top_left,
                       interpolate_point(top_left, top_right, 1 / 3),
                       interpolate_point(lower_left, lower_right, 1 / 3),
                       interpolate_point(top_left, lower_left, 1 / 3)],
        "upper-mid": [interpolate_point(top_left, top_right, 1 / 3),
                      interpolate_point(top_left, top_right, 2 / 3),
                      interpolate_point(top_right, lower_left, 1 / 3),
                      interpolate_point(top_left, lower_right, 1 / 3)],
        "upper-right": [interpolate_point(top_left, top_right, 2 / 3),
                        top_right,
                        interpolate_point(top_right, lower_right, 1 / 3),
                        interpolate_point(top_right, lower_left, 1 / 3),],
        "mid-left": [interpolate_point(top_left, lower_left, 1 / 3),
                     interpolate_point(top_left, lower_right, 1 / 3),
                     interpolate_point(top_right, lower_left, 2 / 3),
                     interpolate_point(top_left, lower_left, 2 / 3)],
        "center": [interpolate_point(top_left, lower_right, 1 / 3),
                   interpolate_point(top_right, lower_left, 1 / 3),
                   interpolate_point(top_left, lower_right, 2 / 3),
                   interpolate_point(top_right, lower_left, 2 / 3)],
        "mid-right": [interpolate_point(top_right, lower_left, 1 / 3),
                      interpolate_point(top_right, lower_right, 1 / 3),
                      interpolate_point(top_right, lower_right, 2 / 3),
                      interpolate_point(top_left, lower_right, 2 / 3)],
        "lower-left": [interpolate_point(top_left, lower_left, 2 / 3),
                       interpolate_point(top_right, lower_left, 2 / 3),
                       interpolate_point(lower_left, lower_right, 2 / 3),
                       lower_left],
        "lower-mid": [interpolate_point(top_right, lower_left, 2 / 3),
                      interpolate_point(top_left, lower_right, 1 / 3),
                      interpolate_point(lower_left, lower_right, 2 / 3),
                      interpolate_point(lower_left, lower_right, 1 / 3)],
        "lower-right": [interpolate_point(top_left, lower_right, 2 / 3),
                        interpolate_point(top_right, lower_right, 2 / 3),
                        lower_right,
                        interpolate_point(lower_left, lower_right, 2 / 3)]
    }
    # x_min, y_min, x_max, y_max = heatmap_rect[0][0], heatmap_rect[0][1], heatmap_rect[2][0], heatmap_rect[2][1]
    # width = x_max - x_min
    # height = y_max - y_min
    #
    # # Define boundaries for 9 regions
    # row_third = height // 3
    # col_third = width // 3
    #
    # region_boundaries = {
    #     "upper-left": (x_min, y_min, x_min + col_third, y_min + row_third),
    #     "upper-mid": (x_min + col_third, y_min, x_min + 2 * col_third, y_min + row_third),
    #     "upper-right": (x_min + 2 * col_third, y_min, x_max, y_min + row_third),
    #     "mid-left": (x_min, y_min + row_third, x_min + col_third, y_min + 2 * row_third),
    #     "center": (x_min + col_third, y_min + row_third, x_min + 2 * col_third, y_min + 2 * row_third),
    #     "mid-right": (x_min + 2 * col_third, y_min + row_third, x_max, y_min + 2 * row_third),
    #     "lower-left": (x_min, y_min + 2 * row_third, x_min + col_third, y_max),
    #     "lower-mid": (x_min + col_third, y_min + 2 * row_third, x_min + 2 * col_third, y_max),
    #     "lower-right": (x_min + 2 * col_third, y_min + 2 * row_third, x_max, y_max)
    # }

    # Initialize a dictionary to store OCR results for each region
    names_in_regions = {key: [] for key in region_boundaries.keys()}

    # Assign OCR names to regions based on their bounding box
    for detection in ocr_results:
        box, (text, score) = detection
        box_x_min, box_y_min = box[0]
        box_x_max, box_y_max = box[2]
        box_center_x = (box_x_min + box_x_max) // 2
        box_center_y = (box_y_min + box_y_max) // 2

        for region_name, boundary in region_boundaries.items():
            # x1, y1, x2, y2 = boundary
            # if x1 <= box_center_x <= x2 and y1 <= box_center_y <= y2:
            if is_point_in_box((box_center_x, box_center_y), boundary):
                names_in_regions[region_name].append(text)
                break

    return names_in_regions


def classify_bounding_boxes_by_color(llm, image, ocr_results, curr_city, max_num=10, verbose=False):
    """
    Classify bounding boxes based on their average color proximity to red or blue.
    """
    def color_distance(c1, c2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    # red_rgb = np.array([180, 29, 46])
    # blue_rgb = np.array([39, 106, 174])
    red_rgb = np.array([255, 0, 0])
    blue_rgb = np.array([0, 0, 255])

    red_set = []
    blue_set = []

    for curr in ocr_results:
        bbox = curr[0]
        text = curr[1][0]
        confidence = curr[1][1]

        if confidence < 0.9:  # Ignore low confidence results
            continue
        if len(text) > 1 and (re.fullmatch(r'(C\d+\s*)+(R\d+\s*)*', text) or re.fullmatch(r'(R\d+\s*)+(C\d+\s*)*', text)):  # Ignore grid indices
            continue

        x_coords = [int(p[0]) for p in bbox]
        y_coords = [int(p[1]) for p in bbox]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        pixel_values = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if 0 <= x < image.width and 0 <= y < image.height:
                    pixel_values.append(image.getpixel((x, y)))

        if not pixel_values:
            continue

        avg_color = np.mean(pixel_values, axis=0)
        distance_to_red = color_distance(avg_color, red_rgb)
        distance_to_blue = color_distance(avg_color, blue_rgb)

        if distance_to_red < distance_to_blue:
            red_set.append((text, distance_to_red))
        elif distance_to_blue < distance_to_red:
            blue_set.append((text, distance_to_blue))

    red_set = [text for text, _ in sorted(red_set, key=lambda x: x[1])]
    blue_set = [text for text, _ in sorted(blue_set, key=lambda x: x[1])]

    red_set = red_set[:min(max_num, len(red_set))]
    blue_set = blue_set[:min(max_num, len(blue_set))]

    red_set_filtered = llm.query_llm(step='filter_names', content={'list': red_set, 'curr_city': curr_city}, assistant=False, verbose=False)
    try:
        red_set_filtered = ast.literal_eval(red_set_filtered)
    except:
        red_set_filtered = red_set

    blue_set_filtered = llm.query_llm(step='filter_names', content={'list': blue_set, 'curr_city': curr_city}, assistant=False, verbose=False)
    try:
        blue_set_filtered = ast.literal_eval(blue_set_filtered)
    except:
        blue_set_filtered = blue_set

    if verbose:
        print('red_set', red_set_filtered)
        print('blue_set', blue_set_filtered)

    return red_set_filtered, blue_set_filtered


def randomly_sample_place_names(red_set, blue_set, invalid_names):
    # Randomly select up to 3 strings from the red_set as the correct answer
    correct_answer = random.sample(red_set, min(3, len(red_set)))

    # Randomly generate 3 incorrect options
    if len(blue_set) < 3:
        blue_set += invalid_names

    incorrect_answers = []
    for _ in range(3):
        # Select up to 3 strings from blue_set
        blue_sample = random.sample(blue_set, min(len(correct_answer), len(blue_set)))
        # # Optionally add 1 string from red_set to make it more challenging
        # if random.choice([True, False]) and red_set:
        #     blue_sample.append(random.choice(red_set))
        random.shuffle(blue_sample)  # Shuffle to mix red and blue selections
        incorrect_answers.append(blue_sample)

    return correct_answer, incorrect_answers


class OCR:
    def __init__(self):
        self.ocr = PaddleOCR(lang='en', show_log=False)  # need to run only once to download and load model into memory

    def draw_result(self, img_path, result):
        result = result[0]
        image = Image.open(img_path).convert('RGB')

        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='Arial.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save('ocr_result.jpg')

    def run_ocr_detection(self, img_path, verbose=False):
        result = self.ocr.ocr(img_path, cls=False)
        self.draw_result(img_path, result)
        heatmap_rect = find_heatmap_region(result[0])

        invaild_lines = []
        invalid_names = []

        for idx in range(len(result)):
            res = result[idx]
            for line_idx, line in enumerate(res):
                if not is_within_heatmap_rect(line[0], heatmap_rect):
                    invaild_lines.append((idx, line_idx))

        if len(invaild_lines) > 0:
            for idx, line_idx in invaild_lines[::-1]:
                invalid_names.append(result[idx][line_idx][1][0])
                del result[idx][line_idx]

        names_in_regions = divide_heatmap_into_regions(heatmap_rect, result[0])

        if verbose:
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    print(line)
        self.draw_result(img_path, result)

        return result[0], names_in_regions, invalid_names


if __name__ == "__main__":
    ocr = OCR()
    img_path = 'heatmap_map.png'
    ocr.run_ocr_detection(img_path, verbose=True)
