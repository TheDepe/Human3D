import numpy as np

import trimesh
import torch

import MinkowskiEngine as ME

from pathlib import Path
from torch.utils.data import Dataset
from trimesh.transformations import rotation_matrix

from typing import Union, Optional

class RealHorsesMPI(Dataset):
    """Dataset to load all .PLY/.OBJ meshes in a directory (recursively)."""

    def __init__(
        self,
        data_path: Union[str, Path],
        file_identifier: Optional[str] = None,
        ext: Optional[Union[str, list[str]]] = (".ply", ".obj"),
        rotations: Optional[list[dict[str, float]]] = None,
        scaling: float = 1.0,
        voxel_size: float = 0.02,
        device: Union[str, torch.device] = "cpu",
    ):
        self.data_path = Path(data_path)
        assert self.data_path.exists(), f"{str(data_path)} does not exist."

        # normalize extensions to list
        if isinstance(ext, str):
            ext = [ext]

        # gather matching files
        all_files = []
        for e in ext:
            all_files += list(self.data_path.rglob(f"*{e}"))

        if file_identifier:
            self.files = [f for f in all_files if file_identifier in f.name]
        else:
            self.files = all_files

        self.files.sort()
        print(f"Found {len(self.files)} matching files.")

        self.rotations = rotations or []
        self.scaling = scaling
        self.voxel_size = voxel_size
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        pcd_file = self.files[index]
        mesh = trimesh.load(pcd_file, process=False)

        # Apply rotations
        mesh = apply_rotations(mesh, self.rotations)

        # Center the points
        if hasattr(mesh, "vertices"):
            mesh.vertices -= mesh.vertices.mean(axis=0)
            points = mesh.vertices
        elif hasattr(mesh, "points"):
            mesh.points -= mesh.points.mean(axis=0)
            points = mesh.points
        else:
            raise AttributeError(f"No vertices/points found in {pcd_file}")

        # Scale
        points *= self.scaling
        mesh.vertices = points if hasattr(mesh, "vertices") else mesh.points

        # Simple features (dummy color/intensity)
        features = np.ones_like(points)

        # Voxelization coordinates
        coords = np.floor(points / self.voxel_size)

        # Combine features and points
        features = np.hstack((features, points))

        # Quantize to sparse tensor
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=torch.from_numpy(coords).contiguous(),
            features=features,
            return_index=True,
            return_inverse=True,
        )

        sample_coordinates = coords[unique_map]
        coordinates = [torch.from_numpy(sample_coordinates).int()]
        sample_features = features[unique_map]
        features_tensor = [torch.from_numpy(sample_features).float()[:, :3]]

        coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features_tensor)
        data = ME.SparseTensor(
            coordinates=coordinates,
            features=features_tensor[0],
            device=self.device,
        )

        return {
            "data": data,
            "coords": coords,
            "features": torch.from_numpy(sample_features).float(),
            "unique_map": unique_map,
            "inverse_map": inverse_map,
            "path": str(pcd_file),
            "original_mesh": mesh,
        }

        

def apply_rotations(mesh: trimesh.Trimesh, rotations: Optional[list[dict[str, float]]] = None):
    """
    Apply sequential axis-angle rotations to a trimesh mesh.

    Args:
        mesh: Trimesh mesh to transform.
        rotations: List of dicts like {"axis": "x", "angle": 90}.
    """
    if not rotations:
        return mesh

    axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

    for rot in rotations:
        axis = rot.get("axis", "z").lower()
        angle = np.radians(rot.get("angle", 0.0))
        if axis not in axis_map:
            raise ValueError(f"Invalid axis '{axis}', must be one of {list(axis_map.keys())}")
        R = rotation_matrix(angle, axis_map[axis])
        mesh.apply_transform(R)

    return mesh
