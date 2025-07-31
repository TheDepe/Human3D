from plyfile import PlyData, PlyElement
import numpy as np
import torch
import hydra
import albumentations as A
import MinkowskiEngine as ME
from trainer.trainer import InstanceSegmentation

from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose



def load_mesh(pcl_file):
    plydata = PlyData.read(pcl_file)
    points = np.stack([plydata['vertex'][axis] for axis in ('x', 'y', 'z')], axis=-1)

    if 'red' in plydata['vertex'].properties:
        colors = np.stack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']], axis=-1)
    else:
        colors = np.full((len(points), 3), 255, dtype=np.uint8)

    return points, colors


def prepare_data(points, colors, device):
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    pseudo_image = colors[np.newaxis, :, :]
   
    coords = np.floor(points / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=torch.from_numpy(coords).contiguous(),
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(coordinates=coordinates, features=features, device=device)

    return data, points, colors, features, unique_map, inverse_map


def map_output_to_pointcloud(outputs, inverse_map, num_vertices, label_space='scannet200', confidence_threshold=0.9):
    logits = outputs["pred_human_logits"][0].detach().cpu()
    masks = outputs["pred_masks"][0].detach().cpu()

    labels_mapped = np.zeros((num_vertices, 1))
    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200 and c > confidence_threshold:
            label_offset = 1 if label_space == 'scannet200' else 0
            labels_mapped[m[inverse_map].numpy()] = int(l) + label_offset

    return labels_mapped


def save_colorized_ply(points, labels, output_file):
    color_map = {
        0: [255, 255, 255],  # background
        1: [255, 0, 0],      # human
    }
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {int(label)}: {count} points")


    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for label in np.unique(labels):
        label = int(label)
        if label in color_map:
            mask = (labels[:, 0] == label)
            colors[mask] = color_map[label]
        else:
            raise ValueError(f"Unsupported label {label}")

    vertices = np.array(
        list(zip(points[:, 0], points[:, 1], points[:, 2], colors[:, 0], colors[:, 1], colors[:, 2])),
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el], text=True).write(output_file)


def get_model(checkpoint_path=None):
    from hydra.experimental import initialize, compose

    with initialize(config_path="conf"):
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path
    cfg.general.experiment_name = "Mask3D_horse_eval"
    cfg.general.project_name = "mask3d_horse_seg"
    cfg.general.num_targets = 16
    cfg.data.num_labels = 16
    cfg.model.num_human_queries = 5
    cfg.model.num_parts_per_human_queries = 16
    cfg.trainer.check_val_every_n_epoch = 1
    cfg.general.topk_per_image = -1
    cfg.model.non_parametric_queries = False
    cfg.trainer.max_epochs = 36
    cfg.data.batch_size = 4
    cfg.data.num_workers = 10
    cfg.general.reps_per_epoch = 1
    cfg.model.config.backbone._target_ = "models.Res16UNet18B"
    cfg.general.train_mode = False
    cfg.general.save_visualizations = True

    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


if __name__ == '__main__':

    model = get_model('/oslab-data-4tb/Github/Human3D/checkpoints/horse_mask.ckpt')
    print(model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ply_file = 'test3.ply'
    points, colors = load_mesh(ply_file)
    data, points, colors_norm, features, unique_map, inverse_map = prepare_data(points, colors, device)

    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    labels = map_output_to_pointcloud(outputs, inverse_map, len(points))

    

    save_colorized_ply(points, labels, 'res.ply')
