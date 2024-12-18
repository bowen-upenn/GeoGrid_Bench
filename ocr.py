from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


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
        for idx in range(len(result)):
            res = result[idx]
            if verbose:
                for line in res:
                    print(line)

        if verbose:
            # draw result
            self.draw_result(img_path, result)
        return result[0]


if __name__ == "__main__":
    ocr = OCR()
    img_path = 'heatmap_map.png'
    ocr.run_ocr_detection(img_path, verbose=True)
