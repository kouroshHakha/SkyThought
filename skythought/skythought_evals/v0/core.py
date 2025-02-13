from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import ray.data
import os
from ray.data.llm import (
    build_llm_processor,
    VLLMProcessorConfig,
    HTTPRequestProcessorConfig,
)

if TYPE_CHECKING:
    from ray.llm._internal.batch.processor import Processor


def apply_processors(ds: ray.data.Dataset, processors: List[Processor]) -> ray.data.Dataset:
    """Apply a list of processors to a dataset"""
    for processor in processors:
        ds = processor(ds)
    return ds




# @dataclass
# class HTTPBackendConfig:
#     """Configuration for HTTP-based model endpoints"""
#     model: str
#     api_url: str
#     api_key: str
#     extra_headers: Optional[Dict[str, str]] = None
#     # HTTP specific scaling params
#     qps: Optional[float] = None
#     concurrency: int = 1
#     batch_size: int = 64

#     def get_ray_llm_batch_config(self) -> HTTPRequestProcessorConfig:
#         """Convert to ray.data.llm processor config"""
#         headers = {"Authorization": f"Bearer {self.api_key}"}
#         if self.extra_headers:
#             headers.update(self.extra_headers)
            
#         return HTTPRequestProcessorConfig(
#             url=self.api_url,
#             header=headers,
#             qps=self.qps,
#             concurrency=self.concurrency,
#             batch_size=self.batch_size,
#         )

# @dataclass 
# class VLLMBackendConfig:
#     """Configuration for vLLM-based models"""
#     model: str
#     engine_kwargs: Optional[Dict[str, Any]] = None
#     # vLLM specific scaling params
#     accelerator_type: Optional[str] = None
#     concurrency: int = 1
#     batch_size: int = 64

#     def get_ray_llm_batch_config(self) -> VLLMProcessorConfig:
#         """Convert to ray.data.llm processor config"""
#         return VLLMProcessorConfig(
#             model=self.model,
#             engine_kwargs=self.engine_kwargs or {},
#             accelerator_type=self.accelerator_type,
#             concurrency=self.concurrency,
#             batch_size=self.batch_size,
#         )

# class Env(ABC):
#     """Base class for evaluation environments"""
    
#     def __init__(self, **kwargs):
#         self.backend_config = None
#         self.template = None
#         self.sampling_params = None
#         self._dataset = None
#         self._llm_processor = None
        
#         # Build the score processors
#         self._score_processors = self._build_score_processors()
    
#     @OverrideToImplementCustomization    
#     def _build_score_processors(self) -> List["Processor"]:
#         return []
    
#     def _preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
#         """Default preprocessor that applies the template"""
#         prompt = self.apply_prompt_template_on_item(item)
#         payload = {
#             "messages": prompt,
#             "sampling_params": self.sampling_params
#         }
        
#         # Add model name for HTTP endpoints
#         if isinstance(self.backend_config, HTTPBackendConfig):
#             payload["model"] = self.backend_config.model
            
#         return payload
        
#     def _postprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
#         """Default postprocessor that extracts generated text"""
#         return {"generated_text": item["generated_text"]}

        
#     @PublicAPI
#     @abstractmethod
#     def read_dataset(self) -> ray.data.Dataset:
#         """Read the default dataset for this environment"""
#         pass
    
#     @PublicAPI
#     def setup(self, 
#               backend_config: Union[VLLMBackendConfig, HTTPBackendConfig],
#               template: Optional[Union[str, Dict, list]] = None,
#               sampling_params: Optional[Dict[str, Any]] = None) -> None:
#         """Configure the environment with backend and generation parameters"""
#         self.backend_config = backend_config
#         self.template = template or self.template
#         self.sampling_params = sampling_params or {}
        
#         # Build the llm processor
#         processor_config = self.backend_config.get_ray_llm_batch_config()
#         self._llm_processor = build_llm_processor(
#             processor_config,
#             preprocess=self._preprocess_item,
#             postprocess=self._postprocess_item,
#         )
        
#     def apply_prompt_template_on_item(self, item: Dict[str, Any]) -> str:
#         """Apply the prompt template to a single item"""
#         if isinstance(self.template, str):
#             return self.template.format(**item)
#         elif isinstance(self.template, list):
#             return [
#                 {"role": msg["role"], "content": msg["content"].format(**item)}
#                 for msg in self.template
#             ]
#         else:
#             raise ValueError(f"Unsupported template type: {type(self.template)}")
        
#     @PublicAPI
#     def generate(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
#         """Run generation on the dataset"""
#         if self._llm_processor is None:
#             raise RuntimeError("Environment not set up. Call setup() first.")
#         return self._llm_processor(dataset)
    
    
#     @PublicAPI
#     def score(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
#         """Score the generations in the dataset"""
#         # iteratively apply the score processors        
#         ds = dataset
#         for score_processor in self._score_processors:
#             ds = score_processor(ds)
#         return ds
    
#     @PublicAPI
#     def save(self, dataset: ray.data.Dataset, path: str, format: str = "json"):
#         """Save the dataset results
        
#         Args:
#             dataset: Dataset to save
#             path: Directory to save results
#             format: Format to save in (json, parquet, etc)
#         """
#         os.makedirs(path, exist_ok=True)
        
#         # Save generations
#         os.makedirs(os.path.join(path, "generations"), exist_ok=True)
#         dataset.write_json(os.path.join(path, "generations"))
        
#         # Save scores if they exist
#         if "score" in dataset.columns():
#             os.makedirs(os.path.join(path, "scores"), exist_ok=True)
#             dataset.select_columns(["score"]).write_json(
#                 os.path.join(path, "scores")
#             )
            
#         # Save failures if they exist
#         failures = dataset.filter(lambda x: "error" in x)
#         if not failures.is_empty():
#             os.makedirs(os.path.join(path, "failures"), exist_ok=True)
#             failures.write_json(os.path.join(path, "failures"))


llm_processor = build_llm_processor(
    HTTPRequestProcessorConfig(
        url="http://localhost:8000/v1/chat/completions",
        header={"Authorization": "Bearer sk-proj-..."},
        qps=10,
        concurrency=10,
        batch_size=10,
    ),
    preprocess=lambda x: {"messages": x["messages"]},
    postprocess=lambda x: {"generated_text": x["choices"][0]["message"]["content"]},
)

score_processors = [
    build_readability_score_processor(...),
    build_math_score_processor(...),
    build_code_score_processor(...),
]


ds = read_dataset()
ds = llm_processor(ds)

ds = ds.map(math_score)
ds = ds.map(code_score)

llm_processor2 = build_llm_processor(
    HTTPRequestProcessorConfig(
        url="http://localhost:8000/v1/chat/completions",
        header={"Authorization": "Bearer sk-proj-..."},
        qps=10,
        concurrency=10,
        batch_size=10,
    ),
    preprocess=lambda x: {"messages": x["messages"]},
    postprocess=lambda x: {"generated_text": x["choices"][0]["message"]["content"]},
)


ds = llm_processor2(ds)
ds = criteria_processor(ds, ...)

for score_processor in score_processors:
    ds = score_processor(ds)

ds.materialize()









