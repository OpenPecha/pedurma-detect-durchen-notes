import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from google.cloud import vision
from google.cloud.vision import AnnotateImageResponse
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import config
from models import double_tsek_detector, note_detector

vision_client = None


def remove_old_maker(image, position):
    ih, iw, _ = image.shape
    x, y, w, h = position
    patch_start_x = iw // 2
    patch = image[0:h, patch_start_x : patch_start_x + w]
    image[y : y + h, x : x + w] = patch
    return image


def insert_text(image, text, position, font_size):
    x, y = position
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    font = ImageFont.truetype(
        "/Library/Fonts/Supplemental/Arial Unicode.ttf", font_size
    )
    draw = ImageDraw.Draw(pil_image)
    draw.text((x, y), text, font=font, fill=(65, 65, 65, 255))
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def alter_note_markers(image, matches):
    radius = int(0.0062 * image.shape[1])
    for x, y, w, h in matches:
        image = remove_old_maker(image, (x, y, w, h))
        x_center = (2 * x + w) // 2
        y_center = (2 * y + h) // 2
        # insert black circle covering the note maker
        cv2.circle(image, (x_center, y_center), radius, (65, 66, 65), -1)
    return image


def alter_double_tseks(image, matches):
    x_adjust = 0.0139 * image.shape[1]
    font_size = 0.0278 * image.shape[1]
    for x, y, w, h in matches:
        image = remove_old_maker(image, (x, y, w, h))
        image = insert_text(image, "【", (x - x_adjust, y), font_size)
    return image


def transform_image(
    input_image_fn: Path, output_image_fn: Path, imshow=False, verbose=False
):
    image = cv2.imread(str(input_image_fn))

    note_matches = list(note_detector.predict(input_image_fn))
    double_tsek_matches = list(double_tsek_detector.predict(input_image_fn))

    if verbose:
        print("Image size:", image.shape)
        print("Notes count:", len(note_matches))
        print("Double Tsek count:", len(double_tsek_matches))

    if imshow:
        image_copy = image.copy()
        for x, y, w, h in note_matches:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for x, y, w, h in double_tsek_matches:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        plt.figure(figsize=(20, 15), dpi=100)
        plt.imshow(image_copy)
        plt.show()

    image = alter_note_markers(image, note_matches)
    image = alter_double_tseks(image, double_tsek_matches)

    if imshow:
        plt.figure(figsize=(20, 15), dpi=100)
        plt.imshow(image)
        plt.show()

    cv2.imwrite(str(output_image_fn), image)


def transform_volume(vol_path, output_dir):
    images_path = vol_path / "images"
    for image_fn in (pbar := tqdm(list(images_path.iterdir()))):
        if not image_fn.name.endswith(".jpg"):
            continue
        pbar.set_description("- " + vol_path.name)
        output_fn = output_dir / image_fn.name
        if output_fn.is_file():
            continue
        transform_image(image_fn, output_fn)


def run_transform():
    for vol_path in (pbar := tqdm(list(config.INPUT_PATH.iterdir()))):
        pbar.set_description("Transform Progress")
        output_dir = config.IMAGES_OUTPUT_PATH / vol_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        transform_volume(vol_path, output_dir)


def google_ocr(image_fn, lang_hint=None):
    global vision_client
    if not vision_client:
        vision_client = vision.ImageAnnotatorClient()
    with io.open(image_fn, "rb") as image_file:
        content = image_file.read()
    ocr_image = vision.Image(content=content)
    features = [
        {
            "type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
            "model": "builtin/weekly",
        }
    ]
    image_context = {}
    if lang_hint:
        image_context["language_hints"] = [lang_hint]

    response = vision_client.annotate_image(
        {"image": ocr_image, "features": features, "image_context": image_context}
    )
    response_json = AnnotateImageResponse.to_json(response)
    response = json.loads(response_json)
    return response


def ocr_volume(vol_path, output_dir):
    output_pages_dir = output_dir / "pages"
    output_combined_fn = output_dir / f"{output_dir.name}.txt"
    combined_pages = ""
    for image_fn in (pbar := tqdm(list(vol_path.iterdir()))):
        if not image_fn.name.endswith(".jpg"):
            continue
        pbar.set_description("- " + vol_path.name)
        output_page_fn = output_pages_dir / f"{image_fn.stem}.txt"
        if output_page_fn.is_file():
            continue
        result = google_ocr(image_fn)
        try:
            text = result["textAnnotations"][0]["description"]
        except:
            continue
        combined_pages += text + "\n\n"
        output_page_fn.write_text(text)
    output_combined_fn.write_text(combined_pages)


def run_ocr():
    for vol_path in (pbar := tqdm(list(config.IMAGES_OUTPUT_PATH.iterdir()))):
        pbar.set_description("OCR Progress")
        output_dir = config.OCR_OUTPUT_PATH / vol_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        ocr_volume(vol_path, output_dir)


if __name__ == "__main__":
    try:
        mode = sys.argv[1]
    except IndexError:
        print("No mode specified. Use 'transform' or 'ocr'")
        sys.exit(1)

    if mode == "transform":
        print("Transforming images...")
        run_transform()
    elif mode == "ocr":
        print("Running OCR...")
        run_ocr()
    else:
        print("Invalid mode. Use 'transform' or 'ocr'")
        sys.exit(1)
