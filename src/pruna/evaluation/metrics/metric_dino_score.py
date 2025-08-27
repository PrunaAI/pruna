import torch
import torch.nn.functional as F
import timm
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.registry import MetricRegistry

@MetricRegistry.register("dino_score")
class DinoScore(StatefulMetric):
    metric_name = "dino_score"

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        # Load the DINO ViT-S/16 model once
        self.model = timm.create_model("dino_vits16", pretrained=True)
        self.model.eval().to(device)
        # Add internal state to accumulate similarities
        self.add_state("similarities", default=[], dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, x, gt, outputs=None):
        """
        x: list of generated images (tensors)
        gt: list of reference images (tensors)
        outputs: optional, ignored
        """
        x = torch.stack(x).to(self.device)
        gt = torch.stack(gt).to(self.device)

        # Extract embeddings ([CLS] token)
        emb_x = self.model.forward_features(x)
        emb_gt = self.model.forward_features(gt)

        # Normalize embeddings
        emb_x = F.normalize(emb_x, dim=1)
        emb_gt = F.normalize(emb_gt, dim=1)

        # Compute cosine similarity
        sim = (emb_x * emb_gt).sum(dim=1)
        self.similarities.append(sim)

    def compute(self):
        sims = torch.cat(self.similarities)
        mean_sim = sims.mean().item()
        return {"name": self.metric_name, "value": mean_sim}
