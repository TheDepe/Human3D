import hydra
import torch

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
        x = self.model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
            clip_feat=clip_feat,
            clip_pos=clip_pos,
        )
        return x
    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

# imports for input loading
import albumentations as A
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d

def get_model(checkpoint_path=None):


    # Initialize the directory with config files
    with initialize(config_path="conf"):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path

    # would be nicd to avoid this hardcoding below
    cfg.general.experiment_name = "Human3D_eval"
    cfg.general.project_name = "human3d"
    cfg.general.num_targets = 16
    cfg.data.num_labels = 16
    # cfg.model = "mask3d_hp"
    # cfg.loss = "set_criterion_hp"
    #cfg.model.num_human_queries = 5
    #cfg.model.num_parts_per_human_queries = 16
    cfg.trainer.check_val_every_n_epoch = 1
    cfg.general.topk_per_image = -1  # Use -1 to indicate no limit or a special behavior
    cfg.model.non_parametric_queries = False
    cfg.trainer.max_epochs = 36
    cfg.data.batch_size = 4
    cfg.data.num_workers = 10
    cfg.general.reps_per_epoch = 1
    cfg.model.config.backbone._target_ = "models.Res16UNet18B"
    cfg.general.checkpoint = checkpoint_path
    cfg.general.train_mode = False
    cfg.general.save_visualizations = True

        
        #TODO: this has to be fixed and discussed with Jonas
        # cfg.model.scene_min = -3.
        # cfg.model.scene_max = 3.

    # # Initialize the Hydra context
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path="conf")

    # Load the configuration
    # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    # for cropping
    # min_bound = np.array([1.27819920, -3.04697800, -1.14556611])
    # max_bound = np.array([4.38896847, 1.98770964, 1.54433572])

    # # Apply cropping
    # in_bounds = (points >= min_bound) & (points <= max_bound)
    # in_bounds = in_bounds.all(axis=1)
    # points = points[in_bounds]
    # colors = colors[in_bounds]
    # mesh.vertices = o3d.utility.Vector3dVector(points)
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh

def prepare_data(mesh, device):
    
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    
    points = np.asarray(mesh.vertices)
    if len(mesh.vertex_colors) == 0:
        # Default color - white
        colors = np.full((len(points), 3), 255, dtype=np.uint8)
    else:
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    
    # fix rotation bug
    # points = points[:, [0, 2, 1]]
    # points[:, 2] = -points[:, 2]

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

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
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    
    
    return data, points, colors, features, unique_map, inverse_map


def map_output_to_pointcloud(mesh, 
                             outputs, 
                             inverse_map, 
                             label_space='scannet200',
                             confidence_threshold=0.9):
    
    # parse predictions
    logits = outputs["pred_human_logits"]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()

    labels = []
    confidences = []
    masks_binary = []

    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200 and c > confidence_threshold:
            labels.append(l.item())
            confidences.append(c.item())
            masks_binary.append(
                m[inverse_map])  # mapping the mask back to the original point cloud
    
    # save labelled mesh
    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles

    labels_mapped = np.zeros((len(mesh.vertices), 1))

    for i, (l, c, m) in enumerate(
        sorted(zip(labels, confidences, masks_binary), reverse=False)):
        
        if label_space == 'scannet200':
            label_offset = 1
            
            l = int(l) + label_offset
                        
        labels_mapped[m == 1] = l
        
    return labels_mapped


def save_colorized_mesh(mesh, labels_mapped, output_file):
    # Define a simple color map for two classes: 0 (background) and 1 (human)
    color_map = {
        0: [255, 255, 255],  # White for background
        1: [255, 0, 0]       # Red for human
    }
    
    # Initialize a color array for all vertices in the mesh
    colors = np.zeros((len(mesh.vertices), 3))
    
    # Get unique labels within the mapped labels
    unique_labels = np.unique(labels_mapped)
    print(unique_labels)
    
    # Apply colors based on the unique labels found in labels_mapped
    for li in unique_labels:
        if li != 0: li=1.0
        if li in color_map:
            # Apply color to vertices where label matches
            colors[(labels_mapped == li)[:, 0], :] = color_map[li]
        else:
            # Handle unexpected label
            raise ValueError(f"Label {li} not supported by the defined color map.")
    
    # Normalize the color values to be between 0 and 1
    colors = colors / 255.0
    
    # Assign colors to mesh vertices
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Write the colorized mesh to the specified output file
    o3d.io.write_triangle_mesh(output_file, mesh)



if __name__ == '__main__':
    
    model = get_model('./checkpoints/human3d.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load input data
    pointcloud_file = './data/raw/egobody/recording_20210910_S05_S06_01_scene_main_01661.ply'
    mesh = load_mesh(pointcloud_file)
    
    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)
    
    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
    
    print(outputs.keys())
    # map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map, confidence_threshold=0.5)
    # save colorized mesh
    save_colorized_mesh(mesh, labels, 'data/pcl_labelled_zed_2.ply')