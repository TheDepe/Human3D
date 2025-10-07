# segmentation_pipeline.py
import torch
import yaml
import trimesh
import tqdm
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any, Dict

import hydra
import MinkowskiEngine as ME
import albumentations as A
from trimesh.transformations import rotation_matrix
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from datasets.real_horses import RealHorsesMPI

# If your dataset module exposes collate_sparse_batch, import it.
# Otherwise uncomment the fallback collate implementation below.
from datasets.real_horses import collate_sparse_batch_inference  # expected signature below

# -------------------------------------------------------------------
# CONFIG HANDLING
# -------------------------------------------------------------------
@dataclass
class ConfigLoader:
    yaml_path: Path

    def load(self) -> dict:
        with open(self.yaml_path, "r") as f:
            return yaml.safe_load(f)


# -------------------------------------------------------------------
# MODEL WRAPPER
# -------------------------------------------------------------------
class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


# -------------------------------------------------------------------
# MODEL BUILDER
# -------------------------------------------------------------------
class ModelBuilder:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    def build(self):
        from hydra.experimental import initialize, compose

        with initialize(config_path="conf"):
            cfg = compose(
                config_name="config_base_instance_segmentation.yaml",
                overrides=["model=mask3d"],
            )

        # Manual overrides (you can move the overrides to YAML later)
        cfg.general.checkpoint = str(self.checkpoint_path)
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

        # Load weights (if present)
        if cfg.general.backbone_checkpoint is not None:
            cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
        if cfg.general.checkpoint is not None:
            cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, cfg, device


# -------------------------------------------------------------------
# MESH PROCESSING UTILITIES
# -------------------------------------------------------------------
class MeshProcessor:
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg = cfg

    # map_output_to_pointcloud now tolerates numpy or torch inverse_map
    def map_output_to_pointcloud(self, mesh: trimesh.Trimesh, outputs: Dict[str, torch.Tensor], inverse_map: Any, threshold: float = 0.5):
        """
        outputs expected to be CPU tensors for a single sample batch (batch dim = 1)
        or already sliced so logits_batch[0] is the correct logits tensor.
        inverse_map: numpy array or torch tensor mapping quantized indices -> original points.
        """
        try:
            logits_batch = outputs["pred_logits"]
            masks_batch = outputs["pred_masks"]
        except KeyError:
            logits_batch = outputs["pred_human_logits"]
            masks_batch = outputs["pred_masks"]

        # logits_batch shape: [1, Q, C]  (we expect batch dim)
        logits = logits_batch.detach().cpu()      # [Q, C]
        masks = masks_batch.detach().cpu().T      # [Q, N_quantized]  -> transposed to [Q, N]

        # Accept inverse_map either numpy array or torch tensor
        if isinstance(inverse_map, np.ndarray):
            inv = inverse_map
        elif isinstance(inverse_map, torch.Tensor):
            inv = inverse_map.cpu().numpy()
        else:
            # fallback: attempt to convert
            try:
                inv = np.array(inverse_map)
            except Exception:
                raise TypeError("inverse_map must be numpy array or torch tensor")

        labels = np.zeros((len(mesh.vertices), 1), dtype=np.uint8)

        p_labels = torch.softmax(logits.float(), dim=-1)  # [Q, C]
        p_masks = torch.sigmoid(masks.float())           # [Q, N_quantized]

        for q in range(p_masks.shape[0]):
            label_id = torch.argmax(p_labels[q]).item()
            conf_label = torch.max(p_labels[q]).item()
            mask_bin = p_masks[q] > 0.5
            # avoid 0 division
            denom = (mask_bin.sum().item() if torch.is_tensor(mask_bin.sum()) else mask_bin.sum())
            conf_mask = float(p_masks[q][mask_bin].sum().item()) / (float(denom) + 1e-8)
            confidence = conf_label * conf_mask

            if confidence > threshold:
                # expand to original points using inverse map
                mask_fullres = mask_bin.numpy()[inv] if isinstance(inv, np.ndarray) else mask_bin.numpy()[inv]
                labels[mask_fullres] = label_id + 1

        return labels

    @staticmethod
    def save_masked_mesh(mesh: trimesh.Trimesh, labels: np.ndarray, output_file: Path, keep_label: int = 1):
        mask = (labels == keep_label)[:, 0]
        faces_keep = mask[mesh.faces].all(axis=1)

        # Filter vertices
        new_vertices = mesh.vertices[mask]

        # Map old vertex indices -> new indices
        old_to_new = -np.ones(len(mask), dtype=int)
        old_to_new[mask] = np.arange(np.sum(mask))

        # Rebuild faces with new indices
        new_faces = old_to_new[mesh.faces[faces_keep]]

        new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
        new_mesh.export(str(output_file))


# -------------------------------------------------------------------
# PRODUCER-CONSUMER PIPELINE
# -------------------------------------------------------------------
import threading
import queue
import time
from datetime import datetime
from torch.utils.data import DataLoader

# -----------------------------
# Helper for simple timestamps
# -----------------------------
def log(msg, prefix="[INFO]"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} {prefix} {msg}")


class SegmentationPipeline:
    def __init__(self, cfg_path: Path, checkpoint: Path, batch_size: int = 4, num_workers: int = 4, max_queue_size: int = 8):
        self.cfg = ConfigLoader(cfg_path).load()
        self.model, self.model_cfg, self.device = ModelBuilder(checkpoint).build()
        self.output_dir = Path(self.cfg.get("out_path", "."))
        self.suffix = self.cfg.get("out_suffix", "_cleaned.ply")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size

        # AMP scaler not strictly necessary for inference, included for clarity
        self.amp_enabled = True

    def run(self):
        log("Initializing dataset and data loader...")

        pre_pro_cfg = self.cfg.get("preprocessing", None)
        if pre_pro_cfg:
            rotations = [
                {"axis": "x", "angle": pre_pro_cfg.get("rotation_x", 0)},
                {"axis": "y", "angle": pre_pro_cfg.get("rotation_y", 0)},
                {"axis": "z", "angle": pre_pro_cfg.get("rotation_x", 0)} 
            ]
        else:
            rotations = None
        ds = RealHorsesMPI(
            data_path=self.cfg.get("dataset_path"),
            file_identifier=self.cfg.get("file_identifier"),
            ext=self.cfg.get("file_extension", ".ply"),
            rotations=rotations,
            scaling=self.cfg.get("data_scaling", 1),
            device=None,  # important: dataset should NOT move tensors to GPU
        )

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,  # this is the number of *samples* collated into one sparse batch
            num_workers=0, # <--- force single-threaded loader to avoid pickling SparseTensor
            collate_fn=lambda batch: collate_sparse_batch_inference(batch, device='cuda'),
            pin_memory=True,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Dataset contains {len(ds.files)} files")

        # Pre-run report
        self._report(ds.files)

        processor = MeshProcessor(self.model, self.device, self.model_cfg)
        result_queue = queue.Queue(maxsize=self.max_queue_size)
        stop_signal = object()

        # Consumer thread(s) - single for clarity; you may spawn more if CPU bound
        consumer_thread = threading.Thread(
            target=self._consumer_loop,
            args=(result_queue, processor, stop_signal),
            daemon=True
        )
        consumer_thread.start()
        log("Consumer thread started.")

        # Producer: iterate DataLoader batches -> run forward -> enqueue per-sample CPU tensors
        start_time = time.time()
        scaler = None
        if self.amp_enabled:
            scaler = torch.cuda.amp.GradScaler(enabled=False)  # GradScaler with enabled=False is a no-op, but keep semantics
        batch_count = 0

        for batch_idx, batch in enumerate(loader, 1):
            batch_count += 1
            # batch is expected to contain keys: "data", "meshes", "inverse_maps", "paths"
            data: ME.SparseTensor = batch["data"]
            meshes: List[trimesh.Trimesh] = batch["meshes"]
            inverse_maps: List[Any] = batch["inverse_maps"]
            paths: List[str] = batch["paths"]
            features: List[Any] = batch["features"]


            # Move sparse batch to device (this is efficient for ME.SparseTensor)
            #data = data.to(self.device)

            log(f"[B{batch_idx}] Running inference on batch (samples={len(meshes)})")
            # Mixed precision context for inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    outputs = self.model(
                        data,
                        point2segment=[torch.zeros(data.coordinates.shape[0], device='cuda')],
                        raw_coordinates=features[:, -3:],
                        is_eval=True,
                        clip_feat=None,
                        clip_pos=False
                    )


            
            for b in range(len(meshes)):
                sample_outputs = {}
                sample_outputs['pred_logits'] = outputs['pred_logits'][b].detach().cpu().clone()
                sample_outputs['pred_masks'] = outputs['pred_masks'][b].detach().cpu().clone()
                
                #sample_outputs = {k: outputs_cpu[k][b:b+1] for k in outputs_cpu.keys()}
                inv_map = inverse_maps[b]
                path = paths[b]
                # Enqueue a tuple for the consumer to handle
                result_queue.put((meshes[b], sample_outputs, inv_map, path))

            log(f"[B{batch_idx}] Enqueued {len(meshes)} samples for postprocessing")

            # If queue full, wait in small sleeps (avoids busy spin)
            while result_queue.full():
                log("Result queue full — waiting for consumer to catch up...")
                time.sleep(0.1)

        # All batches processed: signal consumer(s) to stop
        log("All batches processed. Sending stop signal to consumer...")
        result_queue.put(stop_signal)
        consumer_thread.join()
        total_time = time.time() - start_time
        log(f"Pipeline complete: batches={batch_count}, total_time={total_time:.2f}s")

    # -----------------------------------------------------------
    # Consumer thread: postprocess + save meshes
    # -----------------------------------------------------------
    def _consumer_loop(self, q: queue.Queue, processor: "MeshProcessor", stop_signal):
        log("Consumer thread running...")
        processed = 0
        while True:
            item = q.get()
            if item is stop_signal:
                log("Stop signal received — consumer exiting.")
                break

            mesh, outputs, inverse_map, path = item
            path = Path(path)
            try:
                log(f"[CONSUMER] Postprocessing: {path.name}")
                labels = processor.map_output_to_pointcloud(mesh, outputs, inverse_map)
                output_file = self._output_path(path)
                processor.save_masked_mesh(mesh, labels, output_file)
                log(f"[CONSUMER] Saved: {output_file}")
                processed += 1
            except Exception as e:
                log(f"[ERROR] Failed processing {path.name}: {e}", prefix="[ERROR]")
            finally:
                q.task_done()

        log(f"Consumer finished. Total processed: {processed}")

    # -----------------------------------------------------------
    # Utility: compute output path
    # -----------------------------------------------------------
    def _output_path(self, input_path: Path) -> Path:
        base = Path(input_path).relative_to(self.cfg.get("dataset_path"))
        return self.output_dir / base.with_name(f"{base.stem}{self.suffix}")

    # -----------------------------------------------------------
    # Pre-run Report
    # -----------------------------------------------------------
    def _report(self, files):
        log("Generating report of planned file operations...")
        print("\n=== Segmentation Report ===")
        for f in files:
            out = self._output_path(Path(f))
            flag = " [! OVERWRITE]" if out.exists() else ""
            print(f"{f} -> {out}{flag}")
        print("===========================\n")
        log("Report generation complete.")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
def main():
    pipeline = SegmentationPipeline(
        cfg_path=Path("test_cfg.yaml"),
        checkpoint=Path("/ssd-disk/data_ssd/VAREN/models/Mask3d/fine_turned_horse_model.ckpt"),
        batch_size=12,
        num_workers=4,
        max_queue_size=36,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
