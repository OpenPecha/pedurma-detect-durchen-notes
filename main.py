import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

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


def is_ocred(fn):
    if not config.OCRED_STATUS_PATH.exists():
        return False
    return fn.name in config.OCRED_STATUS_PATH.read_text().split("\n")


def log_ocred(fn):
    fn.unlink()
    with config.OCRED_STATUS_PATH.open("a") as f:
        f.write(fn.name + "\n")


def is_transformed(fn):
    if not config.TRANSFORMED_STATUS_PATH.exists():
        return False
    return fn.name in config.TRANSFORMED_STATUS_PATH.read_text().split("\n")


def log_transformed(fn):
    fn.unlink()
    with config.TRANSFORMED_STATUS_PATH.open("a") as f:
        f.write(fn.name + "\n")


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
    font_file = Path(__file__).parent / "data" / "fonts" / "Arial_Unicode.ttf"
    font = ImageFont.truetype(str(font_file), font_size)
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
    input_image_fn: Path,
    output_image_fn: Path,
    note_matches,
    double_tsek_matches,
    imshow=False,
    verbose=False,
):
    image = cv2.imread(str(input_image_fn))

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


def batch_transform_images(
    input_image_fns: List[Path], output_path: Path, imshow=False, verbose=False
):

    batch_note_matches = list(note_detector.batch_predict(input_image_fns))
    batch_double_tsek_matches = list(
        double_tsek_detector.batch_predict(input_image_fns)
    )

    for input_image_fn, note_matches, double_tsek_matches in zip(
        input_image_fns, batch_note_matches, batch_double_tsek_matches
    ):
        output_image_fn = output_path / input_image_fn.name
        transform_image(
            input_image_fn,
            output_image_fn,
            note_matches,
            double_tsek_matches,
            imshow,
            verbose,
        )


def transform_volume(vol_path, output_dir):
    images_path = vol_path / "images"
    image_files = [
        image_fn
        for image_fn in images_path.glob("*.jpg")
        if not is_transformed(image_fn)
    ]
    batch_size = 10

    with tqdm(total=len(image_files), desc=f"- {vol_path.name}") as pbar:
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i : i + batch_size]
            batch_transform_images(batch, output_dir)
            for image_fn in batch:
                log_transformed(image_fn)
            pbar.update(len(batch))


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

def process_image(image_fn, output_pages_dir, combined_pages, lang_hint=None):
    if not image_fn.name.endswith(".jpg") or is_ocred(image_fn):
        return
    result = google_ocr(image_fn, lang_hint)
    try:
        text = result["textAnnotations"][0]["description"]
    except KeyError:
        text = ""
    except IndexError:
        text = ""
    except Exception as e:
        print(result)
        print(e)
        return
    combined_pages.append(text + "\n\n")
    output_page_fn = output_pages_dir / f"{image_fn.stem}.txt"
    output_page_fn.write_text(text)
    log_ocred(image_fn)

def ocr_volume(vol_path, output_dir, lang_hint=None):
    output_pages_dir = output_dir / "pages"
    output_pages_dir.mkdir(parents=True, exist_ok=True)
    output_combined_fn = output_dir / f"{output_dir.name}.txt"
    combined_pages = []

    image_files = [image_fn for image_fn in vol_path.iterdir() if image_fn.name.endswith(".jpg")]
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_image, image_fn, output_pages_dir, combined_pages, lang_hint) for image_fn in image_files]
        for future in tqdm(futures, desc=f"- {vol_path.name}"):
            future.result()

    combined_pages_text = "".join(combined_pages)
    output_combined_fn.write_text(combined_pages_text)

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
