import json
import numpy as np
import copy
import os


class BOPDetections:
    def __init__(self, detections_path):
        """
        Args:
            detections_path: directory containing all the default detections
        """
        self.set_file_path(detections_path)
        assert os.path.isfile(self.file_path)

    ######## user API ########
    def set_file_path(self, file_path):
        """
        file_path:   an absolute path to a json file (including .json)
        """
        self.file_path = file_path
        self.load_and_sort_data()

    def load_and_sort_data(self):
        f = open(self.file_path)
        dataset = json.load(f)
        self.sorted_json = sorted(dataset, key=lambda x: (x["scene_id"], x["image_id"]))
        self.dets_lookup = {}
        for x in self.sorted_json:
            k = (x["scene_id"], x["image_id"])
            if k not in self.dets_lookup.keys():
                self.dets_lookup[k] = [x]
            else:
                self.dets_lookup[k].append(x)

    def get_detections(self, scene_id, view_id):
        if (scene_id, view_id) not in self.dets_lookup.keys():
            return []
        dets = self.dets_lookup[(scene_id, view_id)]
        return [self.get_mask(x) for x in dets]

    def get_mask(self, x):
        x["segmentation"]["mask"] = self.rle_to_binary_mask(x["segmentation"])
        return x

    def rle_to_binary_mask(self, rle):
        """Converts a COCOs run-length encoding (RLE) to binary mask.
        :param rle: Mask in RLE format
        :return: a 2D binary numpy array where '1's represent the object

        taken from https://github.com/thodan/bop_toolkit/blob/ebe68b99195cf803e1ec798bb9ae11bfe59e3211/bop_toolkit_lib/pycoco_utils.py#L202
        """
        binary_array = np.zeros(np.prod(rle.get("size")), dtype=bool)
        counts = rle.get("counts")

        start = 0
        for i in range(len(counts) - 1):
            start += counts[i]
            end = start + counts[i + 1]
            binary_array[start:end] = (i + 1) % 2

        binary_mask = binary_array.reshape(*rle.get("size"), order="F")
        return binary_mask



###### example usage #######
# file_path = r"C:\Users\mitadm\Documents\CASPER-3D\cosypose_maskrcnn_synt+real\challenge2022-642947_hb-test.json"
# image_id, view_id = (13, 570)

# loader = DataLoaderBop()
# loader.set_file_path(file_path)
# detections = loader.get_detections(image_id, view_id)
# print(detections)
############################
