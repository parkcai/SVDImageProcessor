from process import *

image_path = "images/1.jpg"

color_mode = "RGB"
assert color_mode in ["gray", "RGB", "viridis", "jet"]

process_mode = "energy_ratio"
assert process_mode in ["rank_ratio", "energy_ratio", "direct_rank_num", "original"]
parameter = 0.1


if __name__ == "__main__":
    process(image_path, color_mode, process_mode, parameter)