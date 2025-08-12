import json
import multiprocessing
from hashlib import md5
from pathlib import Path

import numpy as np
import yaml
from fire import Fire
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
from plyfile import PlyData
import pandas as pd
from base_preprocessing import BasePreprocessing
import random

class RealHorseSegmentation(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "/ssd-disk/data_ssd/VAREN/horse_segmentation_evaluation_dataset (Copy)",
        save_dir: str = "./data/horse/processed/",
        modes: tuple = ("validation", "train", "test"),
        n_jobs: int = -1,
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.n_jobs = n_jobs
        self.modes = modes

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError(f"{self.data_dir} <-")
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        all_label_paths = list(self.data_dir.rglob("*_labels.npy"))
        if not all_label_paths:
            raise ValueError(f"No label files found in {self.data_dir}")

        # Shuffle once
        random.seed(42)
        random.shuffle(all_label_paths)

        self.files = {}

        n = len(all_label_paths)

        if set(modes) == {"train"}:
            self.files["train"] = all_label_paths

        elif set(modes) == {"train", "validation"}:
            split_idx = int(n * 0.9)
            self.files["train"] = all_label_paths[:split_idx]
            self.files["validation"] = all_label_paths[split_idx:]

        elif set(modes) == {"train", "test"}:
            split_idx = int(n * 0.9)
            self.files["train"] = all_label_paths[:split_idx]
            self.files["test"] = all_label_paths[split_idx:]

        elif set(modes) == {"train", "validation", "test"}:
            n_test = int(n * 0.1)
            n_val = int(n * 0.1)
            self.files["test"] = all_label_paths[:n_test]
            self.files["validation"] = all_label_paths[n_test:n_test + n_val]
            self.files["train"] = all_label_paths[n_test + n_val:]

        else:
            raise ValueError(f"Unsupported mode combination: {modes}")

    @logger.catch
    def preprocess(self):
        self.n_jobs = (
            multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        )
        for mode in self.modes:
            database = []
            logger.info(f"Tasks for {mode}: {len(self.files[mode])}")
            parallel_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self.process_file)(file, mode)
                for file in self.files[mode]
            )
            print("FILTERED OUT SCENES:")
            for filebase in parallel_results:
                if not isinstance(filebase, str):
                    database.append(filebase)
                else:
                    print(filebase)
            print("====================")
            self.save_database(database, mode)
        self.fix_bugs_in_labels()
        self.joint_database()
        self.compute_color_mean_std(
            train_database_path=(self.save_dir / "train_database.yaml")
        )

    def preprocess_sequential(self):
        for mode in self.modes:
            database = []
            for filepath in tqdm(self.files[mode], unit="file"):
                filebase = self.process_file(filepath, mode)
                database.append(filebase)
            self.save_database(database, mode)
        self.fix_bugs_in_labels()
        self.joint_database()
        self.compute_color_mean_std(
            train_database_path=(self.save_dir / "train_database.yaml")
        )

    def read_plyfile(self, file_path):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(file_path, "rb") as f:
            plydata = PlyData.read(f)
        if plydata.elements:
            return pd.DataFrame(plydata.elements[0].data).values
        
    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """

        # Extracting scene name from filepath
        scene_name = filepath.stem.replace("cleaned_scan_fit", "")
        labels_path = filepath
        scan_path = Path(str(labels_path).replace("labels.npy","raw_scan.ply"))
        filebase = {
            "filepath": scan_path,
            "scene": scene_name,
            "raw_filepath": str(scan_path),
        }
        
        # Read point cloud
        # reading both files and checking that they are fitting
        pcd = self.read_plyfile(str(scan_path))
        coords = pcd[:, :3]
        # fix rotation bug (THIS IS NOT A BUG. THIS IS JUST CHANGING THE CONVENTION)
        # FROM WHAT TO WHAT?
        #coords = coords[:, [0, 2, 1]]
        #coords[:, 2] = -coords[:, 2]

        # Extract channels
        rgb = np.ones_like(pcd)
        labels = np.load(labels_path)
        if labels.ndim == 1:
            labels = labels[:, np.newaxis]
            # duplicate the part labels as instance labels.
            labels[:, 1] = labels[:, 0]

        part_id = labels[:, [0]]  # part id
        instance_id = labels[:, [1]]  # instance id
    
        # Assemble final dataset [x,y,z,r,b,g, part_id, instance_id] [N, 8]
        points = np.hstack((coords, rgb, part_id, instance_id))
        

        # Remap part indices (dataset parts -> model parts)
        

        # Exclude NANs or infs
        if np.isinf(points).sum() > 0:
            # some scenes (scene0573_01_frame_04) got nans
            return scene_name

        # Generate ground truth labels eg 4501 4502 (last digits are humans instance ids)
        gt_part = part_id * 1000 + instance_id
        gt_human = (part_id > 0.0) * 1000 + instance_id

        # Save stuff
        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = (
            self.save_dir / "gt_human" / mode / f"{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_human.astype(np.int32), fmt="%d")
        filebase["gt_human_filepath"] = str(processed_gt_filepath)

        processed_gt_filepath = (
            self.save_dir / "gt_part" / mode / f"{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_part.astype(np.int32), fmt="%d")
        filebase["gt_part_filepath"] = str(processed_gt_filepath)

        return filebase

    def make_instance_database_sequential(
        self,
        train_database_path: str = "./data/processed/train_database.yaml",
        mode="instance",
    ):
        train_database = self._load_yaml(train_database_path)
        instance_database = []
        for sample in tqdm(train_database):
            instance_database.append(self.extract_instance_from_file(sample))
        self.save_database(instance_database, mode=mode)

    @logger.catch
    def make_instance_database(
        self,
        train_database_path: str = "./data/processed/train_database.yaml",
        mode="instance",
    ):
        self.n_jobs = (
            multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        )
        train_database = self._load_yaml(train_database_path)
        instance_database = []
        logger.info(f"Files in database: {len(train_database)}")
        parallel_results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.extract_instance_from_file)(sample)
            for sample in train_database
        )
        for filebase in parallel_results:
            instance_database.append(filebase)
        self.save_database(instance_database, mode=mode)

    def extract_instance_from_file(self, sample_from_database):
        points = np.load(sample_from_database["filepath"])
        labels = points[:, -2:]
        file_instances = []
        for instance_id in np.unique(labels[:, 1]):
            occupied_indices = np.isin(labels[:, 1], instance_id)
            instance_points = points[occupied_indices].copy()
            instance_classes = (
                np.unique(instance_points[:, 9]).astype(int).tolist()
            )

            hash_string = str(sample_from_database["filepath"]) + str(
                instance_id
            )
            hash_string = md5(hash_string.encode("utf-8")).hexdigest()
            instance_filepath = (
                self.save_dir / "instances" / f"{hash_string}.npy"
            )
            instance = {
                "classes": instance_classes,
                "instance_filepath": str(instance_filepath),
                "instance_size": len(instance_points),
                "original_file": str(sample_from_database["filepath"]),
            }
            if not instance_filepath.parent.exists():
                instance_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.save(instance_filepath, instance_points.astype(np.float32))
            file_instances.append(instance)
        return file_instances

    def fix_bugs_in_labels(self):
        pass

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/train_database.yaml",
    ):
        pass

    def save_database(self, database, mode):
        for element in database:
            self._dict_to_yaml(element)
        self._save_yaml(self.save_dir / (mode + "_database.yaml"), database)

    def joint_database(self, train_modes=["train", "validation"]):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(
                self._load_yaml(self.save_dir / (mode + "_database.yaml"))
            )
        self._save_yaml(
            self.save_dir / "train_validation_database.yaml", joint_db
        )

    @classmethod
    def _read_json(cls, path):
        with open(path) as f:
            file = json.load(f)
        return file

    @classmethod
    def _save_yaml(cls, path, file):
        with open(path, "w") as f:
            yaml.safe_dump(
                file, f, default_style=None, default_flow_style=False
            )

    @classmethod
    def _dict_to_yaml(cls, dictionary):
        if not isinstance(dictionary, dict):
            return
        for k, v in dictionary.items():
            if isinstance(v, dict):
                cls._dict_to_yaml(v)
            if isinstance(v, np.ndarray):
                dictionary[k] = v.tolist()
            if isinstance(v, Path):
                dictionary[k] = str(v)

    @classmethod
    def _load_yaml(cls, filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file


if __name__ == "__main__":
    Fire(RealHorseSegmentation)
