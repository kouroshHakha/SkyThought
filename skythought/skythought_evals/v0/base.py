from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import ray.data
import os
from ray.data.llm import (
    build_llm_processor,
    VLLMProcessorConfig,
    HTTPRequestProcessorConfig,
)

@dataclass
class HTTPBackendConfig:
    """Configuration for HTTP-based model endpoints"""
    model: str
    api_url: str
    api_key: str
    extra_headers: Optional[Dict[str, str]] = None
    # HTTP specific scaling params
    qps: Optional[float] = None
    concurrency: int = 1
    batch_size: int = 64

    def get_ray_llm_batch_config(self) -> HTTPRequestProcessorConfig:
        """Convert to ray.data.llm processor config"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.extra_headers:
            headers.update(self.extra_headers)
            
        return HTTPRequestProcessorConfig(
            url=self.api_url,
            header=headers,
            qps=self.qps,
            concurrency=self.concurrency,
            batch_size=self.batch_size,
        )

@dataclass 
class VLLMBackendConfig:
    """Configuration for vLLM-based models"""
    model: str
    engine_kwargs: Optional[Dict[str, Any]] = None
    # vLLM specific scaling params
    accelerator_type: Optional[str] = None
    concurrency: int = 1
    batch_size: int = 64

    def get_ray_llm_batch_config(self) -> VLLMProcessorConfig:
        """Convert to ray.data.llm processor config"""
        return VLLMProcessorConfig(
            model=self.model,
            engine_kwargs=self.engine_kwargs or {},
            accelerator_type=self.accelerator_type,
            concurrency=self.concurrency,
            batch_size=self.batch_size,
        )

class Env(ABC):
    """Base class for evaluation environments"""
    
    def __init__(self, **kwargs):
        self.backend_config = None
        self.template = None
        self.sampling_params = None
        self._dataset = None
        self._processor = None
        
    def setup(self, 
              backend_config: Union[VLLMBackendConfig, HTTPBackendConfig],
              template: Optional[Union[str, Dict, list]] = None,
              sampling_params: Optional[Dict[str, Any]] = None) -> None:
        """Configure the environment with backend and generation parameters"""
        self.backend_config = backend_config
        self.template = template
        self.sampling_params = sampling_params or {}
        
        # Build the processor
        processor_config = self.backend_config.get_ray_llm_batch_config()
        self._processor = build_llm_processor(
            processor_config,
            preprocess=self._preprocess_item,
            postprocess=self._postprocess_item,
        )
    
    def _preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Default preprocessor that applies the template"""
        prompt = self.apply_prompt_template_on_item(item)
        payload = {
            "messages": prompt,
            "sampling_params": self.sampling_params
        }
        
        # Add model name for HTTP endpoints
        if isinstance(self.backend_config, HTTPBackendConfig):
            payload["model"] = self.backend_config.model
            
        return payload
        
    def _postprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Default postprocessor that extracts generated text"""
        return {"generated_text": item["generated_text"]}
        
    @abstractmethod
    def read_dataset(self) -> ray.data.Dataset:
        """Read the default dataset for this environment"""
        pass
        
    @abstractmethod
    def score_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single item"""
        pass

    def score(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Score the generations in the dataset"""
        return dataset.map(self.score_item)
        
    def apply_prompt_template_on_item(self, item: Dict[str, Any]) -> str:
        """Apply the prompt template to a single item"""
        if isinstance(self.template, str):
            return self.template.format(**item)
        elif isinstance(self.template, list):
            return [
                {"role": msg["role"], "content": msg["content"].format(**item)}
                for msg in self.template
            ]
        else:
            raise ValueError(f"Unsupported template type: {type(self.template)}")
        
    def save(self, dataset: ray.data.Dataset, path: str, format: str = "json"):
        """Save the dataset results
        
        Args:
            dataset: Dataset to save
            path: Directory to save results
            format: Format to save in (json, parquet, etc)
        """
        os.makedirs(path, exist_ok=True)
        
        # Save generations
        os.makedirs(os.path.join(path, "generations"), exist_ok=True)
        dataset.write_json(os.path.join(path, "generations"))
        
        # Save scores if they exist
        if "score" in dataset.columns():
            os.makedirs(os.path.join(path, "scores"), exist_ok=True)
            dataset.select_columns(["score"]).write_json(
                os.path.join(path, "scores")
            )
            
        # Save failures if they exist
        failures = dataset.filter(lambda x: "error" in x)
        if not failures.is_empty():
            os.makedirs(os.path.join(path, "failures"), exist_ok=True)
            failures.write_json(os.path.join(path, "failures"))

    def generate(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Run generation on the dataset"""
        if self._processor is None:
            raise RuntimeError("Environment not set up. Call setup() first.")
        return self._processor(dataset)