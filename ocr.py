from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import random
import re
import ast

import utils


def find_row_col_boxes(ocr_results):
    # Extract row and column names
    row_col_indices = []
    for entry in ocr_results:
        box, (text, confidence) = entry
        if re.fullmatch(r'(C\d+\s*)+(R\d+\s*)*', text) or re.fullmatch(r'(R\d+\s*)+(C\d+\s*)*', text):
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

    # Construct the rectangle
    heatmap_rect = [
        [leftmost_col, uppermost_row],  # Top-left
        [rightmost_col, uppermost_row],  # Top-right
        [rightmost_col, lowermost_row],  # Bottom-right
        [leftmost_col, lowermost_row],  # Bottom-left
    ]
    return heatmap_rect


def is_within_heatmap_rect(bbox, heatmap_rect):
    """Check if a bounding box is fully within the heatmap rectangle.
    boxes should have the format of [upper-left, upper-right, lower-right, lower-left] each of which is [col, row]
    """
    # iou = utils.calculate_iou(bbox, heatmap_rect)
    # print('bbox', bbox, 'heatmap_rect', heatmap_rect, 'iou', iou)
    # return iou >= 0.5

    def is_point_in_box(point, box):
        x, y = point
        x_min = min(box[0][0], box[3][0])
        x_max = max(box[1][0], box[2][0])
        y_min = min(box[0][1], box[1][1])
        y_max = max(box[2][1], box[3][1])
        return x_min <= x <= x_max and y_min <= y <= y_max

    for corner in bbox:
        if is_point_in_box(corner, heatmap_rect):
            return True
    return False


def classify_bounding_boxes_by_color(llm, image, ocr_results, curr_city, max_num=10, verbose=False):
    """
    Classify bounding boxes based on their average color proximity to red or blue.
    """
    def color_distance(c1, c2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

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


def randomly_sample_place_names(red_set, blue_set):
    # Randomly select up to 3 strings from the red_set as the correct answer
    correct_answer = random.sample(red_set, min(3, len(red_set)))

    # Randomly generate 3 incorrect options
    incorrect_answers = []
    print('len(correct_answer)', len(correct_answer))
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
        heatmap_rect = find_heatmap_region(result[0])

        invaild_lines = []
        for idx in range(len(result)):
            res = result[idx]
            for line_idx, line in enumerate(res):
                if not is_within_heatmap_rect(line[0], heatmap_rect):
                    invaild_lines.append((idx, line_idx))
        if len(invaild_lines) > 0:
            for idx, line_idx in invaild_lines[::-1]:
                del result[idx][line_idx]

        if verbose:
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    print(line)
            self.draw_result(img_path, result)

        return result[0]


if __name__ == "__main__":
    ocr = OCR()
    img_path = 'heatmap_map.png'
    ocr.run_ocr_detection(img_path, verbose=True)
