import sys

from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from unet.model.eval_model import evaluate_model_from_hdf5

"""
python3 preprocessing-scripts/generate_test_dataset.py wayne-images/test wayne-images/wayne_test_dataset.hdf5
python3 processing-scripts/evaluate_model.py /home/balvisio/repos/NIH-NEI/mouse-image-segmentation/wayne-results/2021-11-17_12_12_10_U-net_wayne_mice_oct/model_epoch52.hdf5 /home/balvisio/repos/NIH-NEI/mouse-image-segmentation/wayne-images/wayne_test_dataset.hdf5 /home/balvisio/repos/NIH-NEI/mouse-image-segmentation/wayne-evaluation/
"""
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 evaluate_model.py <path/to/model/file> </path/to/test/dataset/file> </output/path/>")
        exit(1)

    model_file_path = sys.argv[1]
    test_dataset_file = sys.argv[2]
    output_path = Path(sys.argv[3])

    evaluate_model_from_hdf5(model_file_path, test_dataset_file, output_path)