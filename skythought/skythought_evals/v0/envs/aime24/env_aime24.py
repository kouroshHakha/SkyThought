from typing import List

from ..scores.math_verify.scorer_math_verify import build_math_verify_score_processor, MathVerifyScoreProcessorConfig
from ..env_base import Env
from ray.llm._internal.batch.processor import Processor
from datasets import load_dataset

import ray

class AIME24Env(Env):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.template = [
            {
                "role": "system",
                "content": "Return your final response within \\boxed{{}}."
            },
            {
                "role": "user",
                "content": "{question}"
            }
        ]
        
    def read_dataset(self) -> ray.data.Dataset:
        
        hf_ds = load_dataset(
            path=self.task_config.dataset_path,
            name=self.task_config.dataset_subset,
            split=self.task_config.dataset_split,
            **self.task_config.dataset_kwargs
        )
        
        ray_ds = ray.data.from_huggingface(hf_ds)
        
        # filter out problems that don't contain "2024"
        ray_ds = ray_ds.filter(lambda x: "2024" in x["url"])
        
        return ray_ds
        
    
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