from __future__ import annotations

from cv2 import BFMatcher
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from skimage.metrics import structural_similarity
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, binary_opening

from scipy.ndimage.morphology import binary_fill_holes

import cv2 as cv

from os import listdir

import matplotlib.pyplot as plt
import numpy as np


class ObjectInfo:
    _max_multy: float = 0.7

    def __init__(self, name: str) -> None:
        self.object_name = name
        self.kp = None
        self.des = None
        self.prop = None

    def set_key_point_info(self, kp, des) -> None:
        self.kp = kp
        self.des = des

    def set_prop(self, prop) -> None:
        self.prop = prop

    def match_proporion(self, obj: ObjectInfo) -> float:
        if len(obj.kp) > len(self.kp):
            return obj.match_proporion(self)

        bf = BFMatcher()
        matches = bf.knnMatch(self.des, obj.des, k=2)

        good = [[m] for m, n in matches if m.distance < self._max_multy * n.distance]

        return len(good) / len(matches)

        # matches = [[match[0].queryIdx, match[0].trainIdx] for match in good]


class InterlligentPlacer:
    _objects_folder: str = '.\..\..\\images\\objects\\'
    _min_area: int = 100
    _surface_path: str = 'surface.jpg'

    def __init__(self) -> None:
        self.objects: list = []
        self.cur_image = None
        self.surface = None

        self.rect_prop = None

        for image_path in listdir(self._objects_folder):
            if image_path.find('object') != -1:
                self.objects.append(self._load_object_info(image_path))

        self.surface = cv.cvtColor(cv.imread(f'{self._objects_folder}\\{self._surface_path}'), cv.COLOR_BGR2RGB)

    def check_image(self, image_path: str) -> bool:
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        surface_gray = cv.cvtColor(self.surface, cv.COLOR_RGB2GRAY)

        (_, diff) = structural_similarity(gray, surface_gray, full=True)

        diff_fill_edges = binary_fill_holes(binary_closing(canny(diff, sigma=2), selem=np.ones((10, 10))))
        diff_fill_edges = binary_opening(diff_fill_edges, selem=np.ones((25, 25)))

        labels = sk_measure_label(diff_fill_edges)
        props = regionprops(labels)

        regions = np.array([prop.centroid[0] for prop in props])
        rect_ind = regions.argmin()
        self.rect_prop = props[rect_ind]

        finded_objects = []

        for object_ind in [ind for ind in range(len(regions)) if ind != rect_ind]:
            mask = labels == (object_ind + 1)
            finded_objects.append(self._define_object(image, mask))

        plt.imshow(diff_fill_edges, cmap='gray')
        plt.show()

        return self._is_fit(finded_objects)

    def _is_fit(self, finded_objects) -> bool:
        objects_area = np.sum([obj.prop.area for obj in finded_objects])

        return objects_area < self.rect_prop.area

    def _load_object_info(self, image_filename: str) -> ObjectInfo:
        image = cv.imread(f'{self._objects_folder}\\{image_filename}')
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        prop = self._extract_object(image)

        kp, des = self._find_key_points(image)

        object_info = ObjectInfo(image_filename)
        object_info.set_key_point_info(kp, des)
        object_info.set_prop(prop)

        # dots = np.vstack([np.asarray(k.pt) for k in kp])

        # plt.imshow(image)
        # plt.plot(dots[:, 0], dots[:, 1], 'or', markersize=2)
        # plt.show()

        return object_info

    def _extract_object(self, image):
        image_threshold = self._threshold(image)

        labels = sk_measure_label(image_threshold)
        props = regionprops(labels)

        object_ind = np.array([prop.area for prop in props]).argmin()
        mask = labels == (object_ind + 1)

        image[~mask] = 255

        return props[object_ind]

    def _threshold(self, image):
        img_blur = gaussian(image, sigma=3, multichannel=True)
        img_blur_gray = rgb2gray(img_blur)

        otsu = threshold_otsu(img_blur_gray)
        res_otsu = img_blur_gray <= otsu

        otsu_enclosed = binary_closing(res_otsu, selem=np.ones((15, 15)))
        otsu_enclosed = binary_opening(otsu_enclosed, selem=np.ones((15, 15)))

        return otsu_enclosed

    def _find_key_points(self, image):
        sift = cv.SIFT_create()

        return sift.detectAndCompute(cv.cvtColor(image, cv.COLOR_RGB2GRAY), None)

    def _define_object(self, input_image, mask) -> ObjectInfo:
        image = np.copy(input_image)
        image[~mask] = 255

        object_info = ObjectInfo(None)
        object_info.set_key_point_info(*self._find_key_points(image))

        object_ind = np.array([obj.match_proporion(object_info) for obj in self.objects]).argmax()

        for i, val in enumerate(np.array([obj.match_proporion(object_info) for obj in self.objects])):
            print(f'{self.objects[i].object_name}: {val}')

        print(self.objects[object_ind].object_name)

        print('########################################################')

        plt.imshow(image)
        plt.show()

        return self.objects[object_ind]
