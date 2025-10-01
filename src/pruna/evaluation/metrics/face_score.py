from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult
from pruna.evaluation.metrics.utils import SINGLE, get_call_type_for_single_metric, metric_data_processor
from pruna.evaluation.metrics.registry import MetricRegistry
import torch

# Import FaceScore and ImageReward
from .FaceScore import FaceScore


@MetricRegistry.register("face_score")
class FaceScoreMetric(StatefulMetric):
	"""
	FaceScoreMetric evaluates the quality of generated human faces using the FaceScore reward model.
	Relies on batch-face and image-reward for scoring.
	"""
	metric_name = "face_score"
	higher_is_better = True
	default_call_type = "y"  # Only predictions are needed

	def __init__(self, call_type=SINGLE):
		super().__init__()
		self.call_type = get_call_type_for_single_metric(call_type, self.default_call_type)
		self.face_score = FaceScore('FaceScore')
		self.add_state("total_score", torch.tensor(0.0))
		self.add_state("count", torch.tensor(0))

	def update(self, inputs, ground_truths, predictions):
		# Only predictions are used for FaceScore
		metric_data = metric_data_processor(inputs, ground_truths, predictions, self.call_type)
		images = metric_data[0]  # Should be a batch of image file paths or PIL images
		batch_score = 0.0
		batch_count = 0
		if not isinstance(images, (list, tuple)):
			images = [images]
		for img in images:
			# If img is a PIL image, save to temp file
			if hasattr(img, 'save'):
				import tempfile
				with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
					img.save(tmp.name)
					scores, _, _ = self.face_score.get_reward(tmp.name)
			elif isinstance(img, str):
				scores, _, _ = self.face_score.get_reward(img)
			else:
				continue
			if isinstance(scores, (list, tuple)):
				batch_score += sum(scores)
				batch_count += len(scores)
		self.total_score += batch_score
		self.count += batch_count

	def compute(self):
		if self.count == 0:
			value = 0.0
		else:
			value = self.total_score.item() / self.count.item()
		params = self.__dict__.copy()
		return MetricResult(self.metric_name, params, value)
