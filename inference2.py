
import hydra
import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from trainer.trainer import InstanceSegmentation
import albumentations as A
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
 


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


# Load configuration # Assuming your Hydra config is stored in config.yaml
def main():

    model = get_model('/oslab-data-4tb/Github/Human3D/checkpoints/horse_mask.ckpt')
    # Switch model to evaluation mode
    model.eval()

    # Specify the path of the .ply file
    ply_file_path = "/oslab-data-4tb/Github/Human3D/test3.ply"  # Replace with your actual .ply file path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read and prepare the data from the .ply file
    full_res_coords, original_colors, original_normals = read_ply_file(ply_file_path)

    # Prepare the data in the correct format for the model
    # Example: If your model expects sparse tensor input
    data = prepare_data_for_model(full_res_coords, original_colors, original_normals)

    data, points, colors_norm, features, unique_map, inverse_map = prepare_data(full_res_coords, original_colors, device)

    # Perform inference
    with torch.no_grad():
        output = model.forward(data, raw_coordinates=features, is_eval=True)

    
    print(f"Output type: {type(output)}")
    print(f"Output content: {output}")

    # model.save_visualizations()

    # Post-process the predictions (e.g., extracting masks, scores, and classes)
    #pred_masks = output["pred_masks"][0].detach().cpu().numpy()
    #pred_scores = output["pred_scores"][0].detach().cpu().numpy()
    #pred_classes = output["pred_classes"][0].detach().cpu().numpy()
    labels = map_output_to_pointcloud(output, inverse_map, len(points))
    save_colorized_ply(points, labels, 'res.ply')


    # Visualize or save the results (depending on your use case)
   
    #save_visualization(pred_masks, pred_scores, pred_classes)

    # Optionally: Evaluate the results if ground truth is available
    # If ground truth is available, you can evaluate the performance using IoU or other metrics
    # If you don't have ground truth, you can skip this step
    # evaluate_results(pred_masks, ground_truth_data)

# Function to read data from the .ply file
def read_ply_file(ply_file_path):
    ply_data = PlyData.read(ply_file_path)
    vertex_data = ply_data['vertex']
   
    # Extract the 3D coordinates and other relevant information (like colors and normals)
    full_res_coords = np.array([list(vertex) for vertex in zip(vertex_data['x'], vertex_data['y'], vertex_data['z'])])
    original_colors = np.array([list(vertex) for vertex in zip(vertex_data['red'], vertex_data['green'], vertex_data['blue'])])
    # Optionally extract normals if available
    if 'nx' in vertex_data and 'ny' in vertex_data and 'nz' in vertex_data:
        original_normals = np.array([list(vertex) for vertex in zip(vertex_data['nx'], vertex_data['ny'], vertex_data['nz'])])
    else:
        original_normals = None  # If normals are not available in the .ply file
   
    return full_res_coords, original_colors, original_normals

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


    print("coords min/max:", np.min(coords), np.max(coords))
    print("colors min/max:", np.min(colors), np.max(colors))
    print("colors mean/std:", np.mean(colors, axis=0), np.std(colors, axis=0))
    

    print("Any NaNs in coords?", np.isnan(coords).any()) 
    print("Any NaNs in colors?", np.isnan(colors).any())


    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    

    sample_features = sample_coordinates  # Use XYZ as features
    features = [torch.from_numpy(sample_features).float()]
    features = [(f - f.mean(dim=0)) / (f.std(dim=0) + 1e-5) for f in features]

    features = torch.cat(features, dim=0)

    data = ME.SparseTensor(coordinates=coordinates, features=features, device=device)

    return data, points, colors, features, unique_map, inverse_map
# Function to prepare the data for the model
def prepare_data_for_model(full_res_coords, original_colors, original_normals):
    # Example of converting data into the required format for your model.
    # If your model uses SparseTensor, you might need to use the MinkowskiEngine here.
    # For the sake of simplicity, we'll just return a tensor with the coordinates for now.

    # This might need adjustment based on your model's data format.
    coordinates = torch.tensor(full_res_coords, dtype=torch.float32)  # Convert to tensor
    features = torch.tensor(original_colors, dtype=torch.float32)  # You may need to normalize or process features

    # Create a SparseTensor or whatever your model requires
    data = ME.SparseTensor(coordinates=coordinates, features=features, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   
    return data

# Function to save the visualization using plyfile
def save_visualization(pred_masks, pred_scores, pred_classes):
    # Example of saving a visualization of the predictions as a .ply file.
    # You can modify this to save the predicted masks and class info.
   
    # For now, we'll assume the `pred_masks` are binary (e.g., 0 or 1) and use these as colors for visualization.
    # You can adjust based on the actual format of the predictions.
   
    full_res_coords = np.random.rand(pred_masks.shape[0], 3)  # Replace with actual coordinates (input data)
    original_colors = np.zeros((pred_masks.shape[0], 3))  # Initialize colors to black (modify as needed)

    # Map predicted class to a color (optional)
    for i in range(len(pred_classes)):
        color = [float(pred_classes[i] / max(pred_classes))] * 3  # Simple coloring based on class
        original_colors[i] = color

    # Create a structured array to hold the point cloud data
    vertex = np.array([(full_res_coords[i][0], full_res_coords[i][1], full_res_coords[i][2], *original_colors[i])
                       for i in range(full_res_coords.shape[0])],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])

    # Create a PlyElement to save the point cloud
    vertex_element = PlyElement.describe(vertex, 'vertex')

    # Save the point cloud to a .ply file
    ply_data = PlyData([vertex_element])
    ply_data.write("visualizations/predictions.ply")
    print(f"Saved visualization for predictions")


def save_colorized_ply(points, labels, output_file):
    color_map = {
        0: [255, 255, 255],  # background
        1: [255, 0, 0],      # human
    }
    
    print(np.unique(labels, return_counts=True))


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
    


if __name__ == "__main__":
    main()