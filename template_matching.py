import math
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np

def thresholding(image, th=80, imshow=False):
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,th,255,cv2.THRESH_BINARY_INV)
    if imshow: plt.imshow(thresh, cmap='gray'); plt.show()
    return thresh


def dilation(thresh_img, imshow=False):
    kernel = np.ones((3,85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
    if imshow: plt.imshow(dilated, cmap='gray'); plt.show()
    return dilated

def find_contours(dialted_img):
    (contours, heirarchy) = cv2.findContours(dialted_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)
    return sorted_contours_lines

def get_line_contours(img, imshow=False):
    """Returns text lines."""
    thresh_img = thresholding(img, imshow=imshow)
    dilated_img = dilation(thresh_img, imshow=imshow)
    sorted_contours_lines = find_contours(dilated_img)

    # filter out one text lines
    if imshow: img2 = img.copy()
    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)
        if h < 30: continue
        if imshow: cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)
        yield x, y, w, h
        
    if imshow: plt.imshow(img2); plt.show()


def get_lines(img):
    for x, y, w, h in get_line_contours(img):
        yield img[y:y+h, x:x+w].copy()


def create_template(image_fn, template_name, template_path, n_line, x1, x2, y1, y2):
    img = cv2.imread(str(image_fn))
    lines = list(get_lines(img))
    line = lines[n_line-1]
    plt.imshow(line)
    plt.show()
    template = line[y1:y2, x1:x2]

    template_edged = cv2.Canny(template, 100, 200)
    plt.subplot(121),plt.imshow(template,cmap = 'gray')
    plt.subplot(122),plt.imshow(template_edged, cmap = 'gray')
    plt.show()

    template_fn =  template_path / f"{template_name}.png"
    cv2.imwrite(str(template_fn), template)


def match_template(img, template, imshow=False, multi=False, edged=False):
    """find the given template on the image
    """
    # convert to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, w, h = template.shape[::-1]
    
    if edged:
        candi_img = cv2.Canny(img_gray, 50, 200)
        candi_template = cv2.Canny(template_gray, 50, 200)
    else:
        candi_img = img_gray
        candi_template = template_gray
        
    res = cv2.matchTemplate(candi_img, candi_template, cv2.TM_CCOEFF_NORMED)
    
    if multi:
        threshold = 0.8
        loc = np.where(res >= threshold)
        print(loc)
        matches = list(zip(*loc[::-1]))
    else:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        matches = [max_loc]
    
    if imshow:
        _, w, h = template.shape[::-1]
        for x, y in matches:
            cv2.rectangle(img, (x, y), (x+w, y+h), 255, 2)
        plt.imshow(img)
        plt.show()

    matches = [(x, y, w, h) for x, y in matches]
    return matches


def find_unique_matches(matches):
    def are_points_close(point1, point2, tolerance):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) <= tolerance
    
    unique_matches = []
    tolerance = 10
    
    for current_point in matches:
        if not any(are_points_close(current_point, unique_point, tolerance) for unique_point in unique_matches):
            unique_matches.append(current_point)

    return unique_matches
    
    
def find_matching_templates(image_fn: Path, templates_path: Path, edged=True, imshow=False) -> list[tuple[int, int, int, int]]:
    """Find all positions of the matching templates and filter out duplicates.

    Returns:
        - unique_matches: list of unique matches, i.e., [(x, y, w, h), ...]
    """
           
    img = cv2.imread(str(image_fn)) 
    matches = []
    for template_fn in sorted(templates_path.iterdir()):
        if not template_fn.name.endswith(".png"): continue
        template = cv2.imread(str(template_fn))
        single_matches = match_template(img, template, edged=edged)
        matches.extend(single_matches)
    
    if not matches:
        return None

    unique_matches = find_unique_matches(matches)
    
    if imshow:
        for x, y, w, h in unique_matches:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        plt.figure(figsize=(20, 15), dpi=100)
        plt.imshow(img)
        plt.show()
        
    return unique_matches
    


