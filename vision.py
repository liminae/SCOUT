import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from geometry import profile


def silhouette(image):
    if image is None or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<64:
        raise ValueError('Expected a color image at least 64 pixels wide and tall')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    if np.median(border)>0:
        binary = cv2.bitwise_not(binary)
    size = max(3, min(15, (min(gray.shape)//100)*2+1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No foreground silhouette found')
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)/gray.size
    x, y, w, h = cv2.boundingRect(contour)
    if not .02<=area<=.85 or x<=1 or y<=1 or x+w>=gray.shape[1]-1 or y+h>=gray.shape[0]-1:
        raise ValueError('Use an isolated silhouette on a plain background with a visible border')
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask, cv2.Canny(smooth, 50, 150), contour


def analyze(image):
    mask, edges, contour = silhouette(image)
    metrics, rows = profile(mask)
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (40, 180, 80), 2)
    points = np.column_stack(((rows[:, 1]+rows[:, 2])/2, rows[:, 0])).astype(np.int32)
    cv2.polylines(overlay, [points], False, (20, 100, 230), 2)
    x = points[:, 0]
    y = points[:, 1]
    fit = np.polyval(np.polyfit(y, x, 1), y)
    reference = np.column_stack((fit, y)).astype(np.int32)
    cv2.polylines(overlay, [reference], False, (230, 100, 30), 2)
    metrics['edge_density'] = float(np.count_nonzero(cv2.bitwise_and(edges, mask))/np.count_nonzero(mask))
    return {k: round(v, 6) for k, v in metrics.items()}, overlay


def demo(bend=0):
    image = np.full((400, 320, 3), 245, np.uint8)
    y = np.arange(55, 345)
    t = (y-y[0])/(y[-1]-y[0])
    center = 160+bend*np.sin(2*np.pi*t)
    width = 58+13*np.cos(2*np.pi*t)
    left = np.column_stack((center-width, y)).astype(np.int32)
    right = np.column_stack((center+width, y))[::-1].astype(np.int32)
    cv2.fillPoly(image, [np.concatenate((left, right))], (35, 35, 35))
    return image


def main():
    parser = argparse.ArgumentParser(description='Experimental 2D silhouette analysis; not a medical diagnosis')
    parser.add_argument('image', nargs='?')
    parser.add_argument('--demo', type=float, metavar='BEND')
    parser.add_argument('--output', default='overlay.png')
    args = parser.parse_args()
    if (args.image is None)==(args.demo is None):
        parser.error('Provide either an image or --demo BEND')
    if args.demo is not None:
        if not-50<=args.demo<=50:
            parser.error('Demo bend must be between -50 and 50')
        image = demo(args.demo)
    else:
        image = cv2.imread(args.image)
    try:
        result, overlay = analyze(image)
        if not cv2.imwrite(str(Path(args.output)), overlay):
            raise ValueError('Could not write overlay')
    except (ValueError, cv2.error) as e:
        parser.error(str(e))
    print(json.dumps({'experimental': True, 'metrics': result}, indent=2))


if __name__=='__main__':
    main()
