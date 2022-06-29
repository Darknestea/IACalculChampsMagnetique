import matplotlib.pyplot as plt
from cv2 import bitwise_and, bitwise_or
from nytche import Microscope
from nytche.utilities.cuttingmasks import CuttingMasks

from Utils.constants import SCREEN_SHAPE, SCREEN_POSITION_CORRECTED, SCREEN_SIZE, SCREEN_DPI
from Utils.simulation_wrapper import add_circle_to_data, add_plane_to_data, convert_to_pixel
import PIL.Image as PilIm
import numpy as np


def test_cutting_plane():
    microscope = Microscope("../Data/i2tem.cbor")
    microscope.biprisms['bpc'].inserted = True
    microscope.biprisms['bpc'].voltage = 400
    beam_tree = microscope.beam()
    beam_slice = beam_tree.find(SCREEN_POSITION_CORRECTED)
    masks = {}
    if beam_slice is None:
        return 0

    for node in beam_slice:
        masks[node] = node.cut(SCREEN_POSITION_CORRECTED)
        print(f"Left/right: {node.left_ray(SCREEN_POSITION_CORRECTED)}/{node.right_ray(SCREEN_POSITION_CORRECTED)}")

    rotation = microscope.larmor_angle(SCREEN_POSITION_CORRECTED)
    rotation = np.pi/2

    data = np.zeros(SCREEN_SHAPE + (3,), dtype=np.uint8)
    for mask in masks.values():
        print("\nNew mask:")
        data_temp = np.zeros(SCREEN_SHAPE + (3,), dtype=np.uint8) + 255
        circles = mask[CuttingMasks.CIRCLES]
        half_plans = mask[CuttingMasks.PLANS]
        for circle in circles:
            print("\tCircle:")
            print(f"\t  x: {int(convert_to_pixel(circle['x']))}")
            print(f"\t  y: {int(convert_to_pixel(circle['y']))}")
            print(f"\t  r: {int(min(convert_to_pixel(circle['r']), SCREEN_SIZE*SCREEN_DPI))}, {circle['r']}")
            add_circle_to_data(circle, data_temp, rotation)
            # img = PilIm.fromarray(data_temp)
            # plt.imshow(img)
            # plt.waitforbuttonpress()
        img = PilIm.fromarray(data_temp)
        plt.imshow(img)
        plt.waitforbuttonpress()
        for half_plan in half_plans:
            print("\tHalfPlane:")
            print(f"\t  x: {convert_to_pixel(half_plan['x'])}, {half_plan['x']})")
            print(f"\t  y: {convert_to_pixel(half_plan['y'])})")
            print(f"\t  theta: {half_plan['theta']})")
            add_plane_to_data(half_plan, data_temp, rotation)
            # img = PilIm.fromarray(data_temp)
            # plt.imshow(img)
            # plt.waitforbuttonpress()
        img = PilIm.fromarray(data_temp)
        plt.imshow(img)
        plt.waitforbuttonpress()
        data[:] = bitwise_or(data, data_temp)
    img = PilIm.fromarray(data)
    plt.imshow(img)
    plt.waitforbuttonpress()
    return 0


if __name__ == '__main__':
    test_cutting_plane()
