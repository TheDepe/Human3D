import hydra
import torch
import trimesh
import numpy as np
import albumentations as A
import MinkowskiEngine as ME
from trimesh.transformations import rotation_matrix

from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)

    def forward(
        self,
        x,
        point2segment=None,
        raw_coordinates=None,
        is_eval=True,
        clip_feat=None,
        clip_pos=None,
    ):
        return self.model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
            clip_feat=clip_feat,
            clip_pos=clip_pos,
        )


from hydra.experimental import initialize, compose

def get_model(checkpoint_path=None):
    # Initialize Hydra config
    with initialize(config_path="conf"):
        cfg = compose(config_name="config_base_instance_segmentation.yaml",
                      overrides=["model=mask3d"])

        
    cfg.general.checkpoint = checkpoint_path
    cfg.general.experiment_name = "Human3D_eval"
    cfg.general.project_name = "human3d"
    cfg.general.num_targets = 3
    cfg.data.num_labels = 2
    cfg.model.num_classes = 2
    cfg.model.num_queries = 1
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
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file, rotate=False, scale=False):
    """Load mesh with Trimesh"""
    mesh = trimesh.load(pcl_file, process=False)

    #mesh.vertices -= mesh.vertices.mean(0)
    if scale:
    # Scale down (meters)
        mesh.vertices *= 1/1000

    if rotate:
        # Rotate X by 90 degrees
        angle_x = np.radians(90)
        R_x = rotation_matrix(angle_x, [1, 0, 0])  # rotation about X
        mesh.apply_transform(R_x)

        # Rotate Z by 90 degrees
        angle_z = np.radians(90)
        R_z = rotation_matrix(angle_z, [0, 0, 1])  # rotation about Z
        mesh.apply_transform(R_z)

    points = mesh.vertices
    # If no vertex colors, assign white
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) > 0:
        mesh.visual.vertex_colors = np.ones_like(mesh.visual.vertex_colors)
        colors = mesh.visual.vertex_colors[:, :3]  # drop alpha if present
    else:
        colors = np.full((len(points), 3), 1, dtype=np.uint8)
    return mesh, points, colors


def prepare_data(points, colors, device):
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / 0.02)
    features = colors

    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], points))
    else:
        features = np.hstack((features, points))
    
    
    use_color = False
    if not use_color:
        features[:, :3] = 1.0  

    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=torch.from_numpy(coords).contiguous(),
        features=features,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = features[unique_map]
    features2 = [torch.from_numpy(sample_features).float()[:, :3]]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features2)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features2[0],
        device=device,
    )

    features = torch.from_numpy(sample_features).float()
    return data, coords, features, unique_map, inverse_map


def map_output_to_pointcloud(mesh, outputs, inverse_map, confidence_threshold=0.5):
    try:
        logits = outputs['pred_logits'][0].detach().cpu()   # [Q, C]
        masks  = outputs['pred_masks'][0].detach().cpu()    # [Q, N]
    except KeyError:
        logits = outputs['pred_human_logits'][0].detach().cpu()
        masks  = outputs['pred_masks'][0].detach().cpu()

    num_queries, num_points = masks.shape
    labels = np.zeros((len(mesh.vertices), 1))

    
    p_labels = torch.softmax(logits, dim=-1)       # [C]
    p_masks  = torch.sigmoid(masks)                # [N]

    l = torch.argmax(p_labels).item()
    c_label = torch.max(p_labels).item()

    # binary mask
    m = p_masks > 0.5
    # mask confidence
    c_m = p_masks[m].sum() / (m.sum() + 1e-8)
    c = c_label * c_m

    if l < 200 and c > confidence_threshold:
        mask_fullres = m[inverse_map.cpu().numpy()]   # expand to full resolution
        labels[mask_fullres] = l + 1   # offset to match dataset
    return labels


def save_colorized_mesh(mesh, labels_mapped, output_file):
    color_map = {
        0: [255, 255, 255],  # background
        1: [255, 0, 0],      # human
    }

    colors = np.zeros((len(mesh.vertices), 3))
    for li in np.unique(labels_mapped):
        li_int = 1 if li != 0 else 0
        colors[(labels_mapped == li)[:, 0], :] = color_map[li_int]

    colors = colors / 255.0
    mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
    mesh.export(output_file)
    print(f"Saved file to {output_file}")


if __name__ == "__main__":
    model = get_model("/ssd-disk/data_ssd/VAREN/models/Mask3d/fine_turned_horse_model.ckpt")
    #model = get_model("/ssd-disk/data_ssd/VAREN/models/Mask3d/horse_mask_synthetic.ckpt")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # WORKS
    pointcloud_file = "/home/dperrett/Documents/horse_project/Repos/Human3D/saved/Mask3D_horse_big_run_eval/visualizations/H0175_still1.000023_full_instance_colored.ply"
    # 
    #pointcloud_file = "/home/dperrett/Documents/horse_project/Repos/Human3D/data/LUCID_HLT003S-001_212300601__20220608164720234_image263.ply"
    #pointcloud_file = "/home/dperrett/Documents/horse_project/Repos/Human3D/data/H0022_still1.000003.obj"
    #pointcloud_file = "/ssd-disk/data_ssd/VAREN/horse_segmentation_evaluation_dataset/H0022/H00022_still1.000051_raw_scan.ply"
    #pointcloud_file = "/ssd-disk/data_ssd/VAREN/horse_segmentation_evaluation_dataset/H0175/H0175_still1.000023_raw_scan.ply"
    pointcloud_file = "/ssd-disk/data_ssd/VAREN/horse_segmentation_evaluation_dataset/H0120/H0120_other1.000047_raw_scan.ply"
    #pointcloud_file = "/home/dperrett/Documents/horse_project/Repos/Human3D/saved/Mask3D_horse_big_run_eval/visualizations/H0120_other1.000047_full_instance_colored.ply"
    
    mesh, points, colors = load_mesh(pointcloud_file, rotate=False, scale=False) # For manually processed data, set these to false

    data, coords, features, unique_map, inverse_map = prepare_data(points, colors, device)

    with torch.no_grad():
        outputs = model(
            data,
            point2segment=[torch.zeros(data.coordinates.shape[0], device=device)],
            raw_coordinates=features[:, -3:],
            is_eval=True,
            clip_feat=None,
            clip_pos=False)
        # x = self.model(
        #         x,
        #         point2segment,
        #         raw_coordinates=raw_coordinates,
        #         is_eval=is_eval,
        #         clip_feat=clip_feat,
        #         clip_pos=clip_pos,
        #     )

    labels = map_output_to_pointcloud(mesh, outputs, inverse_map, confidence_threshold=0.5)
    if len(np.unique(labels)) == 1:
        print("probably something went wrong. Detected no labels.")
    save_colorized_mesh(mesh, labels, "data/pcl_labelled_trimesh.ply")