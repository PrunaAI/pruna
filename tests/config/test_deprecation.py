import pytest
from packaging import version
import pruna
from pruna import SmashConfig

@pytest.mark.parametrize(
    "config, removal_version",
    [
        (dict(dict(whisper_s2t_batch_size=2)), "0.2.8"),
        (dict(dict(ifw_batch_size=2)), "0.2.8"),
        (dict(dict(higgs_example_batch_size=2)), "0.2.8"),
        (dict(dict(diffusers_higgs_example_batch_size=2)), "0.2.8"),
        (dict(dict(torch_compile_batch_size=2)), "0.2.8"),
        (dict(dict(diffusers_int8_enable_fp32_cpu_offload=True)), "0.2.8"),
        (dict(dict(llm_int8_enable_fp32_cpu_offload=True)), "0.2.8"),
        (dict(dict(torch_structured_calibration_samples=2)), "0.2.8"),
        (dict(dict(torch_compile_max_kv_cache_size=2)), "0.2.8"),
        (dict(dict(torch_compile_seqlen_manual_cuda_graph=True)), "0.2.8"),
    ]
)
def test_hyperparameter_warns(config, removal_version):
    smash_config = SmashConfig()
    with pytest.warns(DeprecationWarning, match=f"is deprecated and will be removed in v{removal_version}."):
        smash_config.load_dict(config)
    compare_versions(pruna.__version__, removal_version)


def test_max_batch_size_warns():
    with pytest.warns(DeprecationWarning, match="is deprecated and will be removed in v0.2.8."):
        SmashConfig(max_batch_size=1)
    compare_versions(pruna.__version__, "0.2.8")


def test_prepare_saving_warns():
    config = SmashConfig()
    with pytest.warns(DeprecationWarning, match="is deprecated and will be removed in v0.2.8."):
        config._prepare_saving = True
    compare_versions(pruna.__version__, "0.2.8")


def compare_versions(current_version, deprecation_version):
    current_version = version.parse(current_version)
    deprecation_version = version.parse(deprecation_version)
    assert current_version < deprecation_version, (
        f"old_function() should be removed; current version is {pruna.__version__}"
    )
