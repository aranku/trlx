from typing import Callable

# Register load pipelines via module import
from trlx.pipeline import _DATAPIPELINE
from trlx.pipeline.offline_pipeline import PromptPipeline

# Register load trainers via module import
from trlx.trainer import _TRAINERS, register_trainer
from trlx.trainer.accelerate_ilql_trainer import AccelerateILQLTrainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.trainer.accelerate_odt_trainer import AccelerateODTTrainer
from trlx.trainer.accelerate_sft_trainer import AccelerateSFTTrainer

try:
    from trlx.trainer.nemo_ilql_trainer import NeMoILQLTrainer
except ImportError:
    # NeMo is not installed
    def _trainer_unavailble(name):
        def log_error(*args, **kwargs):
            raise ImportError(f"Unable to import NeMo so {name} is unavailable")

        return register_trainer(name)(log_error)

    _trainer_unavailble("NeMoILQLTrainer")


def get_trainer(name: str) -> Callable:
    """
    Return constructor for specified RL model trainer
    """
    name = name.lower()
    if name in _TRAINERS:
        return _TRAINERS[name]
    else:
        raise Exception("Error: Trying to access a trainer that has not been registered")


def get_pipeline(name: str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")
