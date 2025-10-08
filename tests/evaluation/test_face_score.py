import pytest
import os
from pruna.evaluation.metrics.metric_face_score import FaceScoreMetric
from PIL import Image

@pytest.mark.cpu
def test_face_score_metric_on_sample_image(tmp_path):
	"""
	Test FaceScoreMetric on a sample image (using a blank or local image for simplicity).
	"""
	# Create a blank image for testing
	img = Image.new('RGB', (256, 256), color='white')
	img_path = tmp_path / "test_face.png"
	img.save(img_path)

	metric = FaceScoreMetric()
	# Only predictions are used, so pass the image path as predictions
	metric.update(None, None, [str(img_path)])
	result = metric.compute()
	# The result should be a MetricResult and value should be a float (may be 0 if no face detected)
	assert hasattr(result, 'result')
	assert isinstance(result.result, float)
