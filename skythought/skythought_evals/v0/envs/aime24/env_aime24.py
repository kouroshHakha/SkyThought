from typing import List

from ..scores.math_verify.scorer_math_verify import build_math_verify_score_processor, MathVerifyScoreProcessorConfig
from ..env_base import Env
from ray.llm._internal.batch.processor import Processor
from datasets import load_dataset

import ray

class AIME24Env(Env):
    
    def read_dataset(self) -> ray.data.Dataset:
        
        hf_ds = load_dataset(
            path=self.task_config.dataset_path,
            name=self.task_config.dataset_subset,
            split=self.task_config.dataset_split,
            **self.task_config.dataset_kwargs
        )
        
        
        
        
    
    def _build_score_processors(self) -> List[Processor]:
        
        return [
            build_math_verify_score_processor(
                MathVerifyScoreProcessorConfig(
                    ...
                ),
                preprocess=lambda x: {"generated_text": x["generated_text"]},
                postprocess=lambda x: {**x, "math_score": x["score"]},
            ),
        ]