# Batching

*The code organization and API design have reached an agreement internally between [Cody Yu](mailto:cody@anyscale.com)[Kourosh Hakhamaneshi](mailto:kourosh@anyscale.com)[Praveen Gorthy](mailto:praveeng@anyscale.com)[Hao Chen](mailto:chenh@anyscale.com)*

# Usage Examples

Before diving into the API design, this section provides a set of examples from users’ perspective.

## Minimal Example (Quick Start)

The minimum quick start example (arguments with default values are omitted).

```py
import ray
from ray.data.llm import build_llm_processor, VLLMProcessorConfig

processor_config = VLLMProcessorConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
)
processor = build_llm_processor(
    processor_config,
    preprocess=lambda row: dict(
        messages=row["question"],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=250,
        )
    )
    postprocess=lambda row: dict(
        answer=row["generated_text"]
    ),
    accelerator_type="L40S",
    concurrency=4,
)

# Chain the pipeline.
ds = ray.data.read_parquet(...)
ds = processor(ds)
ds.write_parquet(...)
```

## LLM data pipeline with RAG

A case that leverages LLMs to answer a set of questions with retrieved information from a vector database. The end to end data pipeline includes:

1. A stage that uses an embedding model to generate embedding vectors.  
2. A stage that queries a vector database using the generated embedding vectors.  
3. A stage that combines questions and the retrieved information, and uses LLM to answer the questions.

```py
import ray
from ray.data.llm import (
    build_llm_processor,
    VLLMProcessorConfig,
    HTTPRequestProcessorConfig,
)

# Configure processors.
embed_processor_config = VLLMProcessorConfig(
    model="google-bert/bert-base-uncased",
    task_type="embed", # This is an embedding task.
    engine_kwargs=dict(
        override_pooler_config=dict(...),
    ),
)
retrieve_processor_config = HTTPRequestProcessorConfig(
    url="http://",
    header="...",
    # Control query rates to avoid CDN rate limit.
    qps=1,
)
llm_processor_config = VLLMProessorConfig(
    model="meta-llama/Llama-3.1-70B-Instruct",
    engine_kwargs=dict(pipeline_parallel_size=2),
)

# Create input dataset.
ds = ray.data.read_parquet("...")

# The pipeline to generate embeddings for each row.
ds = build_llm_processor(
    embed_processor_config,
    preprocess=lambda row: dict(
        prompt=row["question"],
        pooling_params=None,
    ),
    postprocess=lambda row: dict(embedding=row["embedding"]),
    accelerator_type="L4",
    concurrency=2,
)(ds)

# Use the embedding to query the vector DB.
ds = build_llm_processor(
    retrieve_processor_config,
    preprocess=lambda row: dict(body=row["embedding"]),
    postprocess=lambda row: dict(retrieved=row["text"]),
    concurrency=4,
)(ds)

# Ask LLM.
ds = build_llm_processor(
    llm_processor_config,
    preprocess=lambda row: dict(
        # Here we combine "question" and "retrieved" to generate LLM inputs.
        messages=[
            {"role": "system", "content": "..."},
            {"role": "user", "content": row["question"] + "\n" + row["retrieved"]},
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=250,        )
    ),
    accelerator_type="L40S",
    concurrency=4,
)(ds)

# Write results
ds.write_parquet("s3://...")
```

## LLM data pipeline with branches

In this example, we construct a data pipeline to let LLMs answer a set of questions. The interesting part of this data pipeline is we first use a small model to judge whether a question is easy or hard, and continue using the small model to answer the easy questions to save cost. The concrete steps include:

1. Use a small model to classify the question (easy, or hard).  
2. Use the same small model to answer the easy questions.  
3. Use the large model to answer the hard questions.

```py
import ray
from ray.data.llm import build_llm_processor, VLLMProcessorConfig

# Create a config for the small model here and use it to create 2 processors.
config_with_small_model = VLLMProcessorConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    engine_kwargs=dict(...),
)
config_with_large_model = VLLMProcessorConfig(
    model="meta-llama/Llama-3.1-70B-Instruct",
    engine_kwargs=dict(pipeline_parallel_size=2),
)

# Create input dataset.
ds = ray.data.read_parquet("...")

# Processor 1: LLM judge to classify the difficulty of each task.
ds = build_llm_processor(
    config_with_small_model,
    preprocess=lambda row: dict(
        messages=[
            {"role": "system",
             "content": "Judge whether the question is easy to answer or not."},
            {"role": "user", "content": row["question"]},
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=50,
            guided_choice=["easy", "hard"],
        )
    ),
    postprocess=lambda row: dict(
        is_easy_task=row["generated_text"] == "easy"
    ),
    accelerator_type="L40S",
    concurrency=2,
)(ds)

# Processor 2: Let the small model deal with easy tasks.
ds_easy_task = ds.filter(lambda row: row["is_easy_task"])
ds_easy_task_answer = build_llm_processor(
    config_with_small_model,
    preprocess_fn=lambda row: dict(
        messages=[
            {"role": "system", "content": ...},
            {"role": "system", "content": ... row["question"] ...},
        ],
        sampling_params=dict(...),
    ),
    postprocess_fn=lambda row: dict(answer=row["generated_text"]),
    accelerator_type="L40S",
    concurrency=2,
)(ds_easy_task)

# Processor 3: Let the large model deal with hard tasks.
ds_hard_task = ds.filter(lambda row: not row["is_easy_task"])
ds_hard_task_answer = build_llm_processor(
    config_with_large_model,
    preprocess_fn=lambda row: dict(
        messages=[
            {"role": "system", "content": ...},
            {"role": "system", "content": ... row["question"] ...},
        ],
        sampling_params=dict(...),
    ),
    postprocess_fn=lambda row: dict(answer=row["_outputs"]["generated_text"]),
    accelerator_type="L40S",
    concurrency=2,
)(ds_hard_task)

# Write results
ds_easy_task_answer.write_parquet("s3://...")
ds_hard_task_answer.write_parquet("s3://...")
```

## LLM data pipeline with online endpoints

A case where users want to run batch inference with public endpoints (e.g., OpenAI).

```py
import ray
from ray.data.llm import build_llm_processor, HTTPRequestProcessorConfig

ds = ray.data.read_parquet("...")

ds = build_llm_processor(
    HTTPRequestProcessorConfig(
        url="https://api.openai.com/v1/chat/completions",
        header="Authorization: Bearer $OPENAI_API_KEY",
        qps=1,
    ),
    preprocess=lambda row: dict(
        model="gpt-4o-mini",
        messages=row["messages"],
        sampling_params=dict(
            temperature=0.0,
            max_tokens=150,
        ),
    ),
    postprocess=lambda row: dict(
        resp=row["generated_text"]
    ),
    concurrency=8,
)

ds.write_parquet("...")
```

#### \[Advanced\] Customize configurations for a particular pipeline stage 

```py
from ray.data.llm import build_llm_processor, VLLMProcessorConfig

# Customize function.
def custom_stage_config(name: str, stage: Stage):
    if name.startswith("TokenizeStage"):
        stage.map_kwargs.update(dict(concurrency=2))

config = VLLMProcessorConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
)

pipeline = build_llm_processor(
    config,
    # Customization
    override_stage_config_fn=custom_stage_config,
    preprocess=lambda row: dict(
        messages=row["question"],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=250,
        )
    )
    postprocess=lambda row: dict(
        answer=row["generated_text"]
    ),
    accelerator_type="L40S",
    concurrency=4,
)
```

# API Design (Implementation Details)

Our API is built on Ray Dataset APIs with the following building blocks (tentative naming):

* **Stage (internal facing. Not exposed to the user):** A wrapper of `ray.data.{map, map_batches}` with optimized user defined function (UDF) for LLMs, such as tokenizer, chat template application, multi-modality input processing, vLLM engine, etc.  
* **Processor (user API):** The unified processor interface. A processor is composed of a sequence of Stages in a certain order to achieve a well-defined functionality for LLM batch inference, such as question answering, retrieval augment generation (RAG), model evaluation, AI agent, etc. In addition to the functionality, Processors are capable of achieving high throughput with optimal Ray Data API configuration (e.g., batch size, max\_concurrency, etc). We use a ProcessorConfig to configure a processor, and use `build_llm_processor` to construct a processor.  
* **ProcessorConfig (user API):** A processor configuration is used to configure a processor, including its global configuration such as checkpointing and batch size. We will have several predefined processor configurations, such as LLMProcessorConfig and HTTPRequestProcessorConfig, for users to configure a certain processor. The default values of each ProcessorConfig are manually tuned to achieve decent throughput, but they can be further optimized for different datasets by the Tuner in the future.  
* **build\_llm\_processor (user API):** The unified API to build a processor based on a particular processor configuration. Internally we register each processor configuration to a corresponding build function to construct a processor for certain functionality.

Here are the high level design principles:

* **One processor only includes at most *one* LLM engine.** If a user workload needs more than one LLM engine, users should build multiple processors and chain them together.  
* **A Processor is composed of a sequence of stages.** We provide decent configurations for a processor to achieve the end-to-end best throughput, but advanced users are able to override any configuration of any stage if needed (see the advanced example).  
* **A processor has a well-defined functionality**, so we do not expect users to change the stage order of a processor. For example, LLM batch inference always applies a chat template, followed by tokenizer, LLM inference, and detokenizer. It makes sense (and configurable) if users want to skip a certain stage (e.g., detokenizer), but it doesn’t make sense to change the order. In other words, if we cannot well define the stage order of a processor, it means we should not introduce this processor at this moment because the desired functionality isn’t clear or widely accepted.

Note: There is the concept of [preprocessor](https://docs.ray.io/en/latest/data/api/doc/ray.data.preprocessor.Preprocessor.html#ray.data.preprocessor.Preprocessor) in ray.data that we can potentially reuse as the base class. However, based on a discussion between Praveen, Hao, Kourosh and Cody, the definition of processor is different from ray.data.Preprocessor. So we decided to define the processor in ray.llm.\_internal.batch for now. We can move this class to ray.data in the future if needed.

In the rest of this section, we introduce the APIs of each building block.

## Stage

Stage is an internal API and we don’t expect most users to know it.

```py
class Stage(BaseModel):
    # A well-optimized stateful UDF for this stage.
    fn: UserDefinedFunction
    # The keyword arguments of the UDF constructor.
    fn_constructor_kwargs: Dict[str, Any]
    # Whether to use .map() or .map_batches(). The default is True.
    batching: bool = True
    # The arguments of .map() or .map_batches().
    map_kwargs: Dict[str, Any]
```

## Processor, ProcessorConfig, and build\_llm\_processor

### Overview

### Data Flow of a Processor

### Interface

```py
# This will be inherited to configure different processors
# (see VLLMProcessorConfig in next section for examples).
class ProcessorConfig(BaseModel):
    # Control the fault tolerance granularity.
    batch_size: int = 64

# A unified builder to create a processor based on different configurations.
class ProcessorBuilder:
    """Build a processor based on the configuration."""
    _registry: Dict[Type[ProcessorConfig], Callable] = {}

    @classmethod
    def register(cls, config_type: Type[ProcessorConfig], builder: Callable):
        """A decorator to assoicate a particular pipeline config
        with its build function.
        """

    @classmethod
    def build(
        cls,
        config: ProcessorConfig,
        override_stage_config_fn: Optional[Callable] = None,
        **kwargs
    ) -> Processor:
        """Build a processor.

        Args:
            config: The processor config.
            override_stage_config_fn: Custom stages configurations.

        Returns:
            The built processor.
        """
        config_type = type(config)
        procesor = cls._registry[config_type](config, **kwargs)
        if override_stage_config_fn is not None:
            for name, stage in procesor.stages.items():
                override_stage_config_fn(name, stage)
        return procesor

class Processor:
    # The reserved column name for the core worker.
    # If users really need to change this name, they need to
    # inherit the pipeline to customize (not the best practice).
    data_column: str = "__data"

    def __init__(self,
        config: ProcessorConfig,
        # Preprocess inputs to fit the processor inputs.
        # The returned dict will be wrapped by the input column, such as
        # {"_inputs": preprocess_fn(row)}.
        preprocess_fn: Optional[Callable[dict, dict]] = None,
        # Postprocess outputs from the processor.
        postprocess_fn: Optional[Callable[dict, dict]] = None,
        # The following are the configurations for the LLM stage,
        # and these are the most important configurations we want users
        # to pay attention in the first place. These configurations will
        # be mapped to the .map_batches() arguments of the corresponding
        # Stage.
        accelerator_type: Optional[str] = None,
        concurrency: int = 1,
    ):
        ...
        self.stages: Dict[str, PipelineStage] = {}

    def __call__(self, dataset: Dataset) -> Dataset:
        """Execute the processor:
        preprocess -> stages -> postprocess.
        Note that the dataset won't be materialized during the execution.

        Args:
            dataset: The input dataset.

        Returns:
            The output dataset.
        """
        ds = ds.map(self.preprocess, ...)
        for stage in self.stages.values():
            if stage.batching:
               ds = ds.map_batches(
                   stage.fn, stage.fn_constructor_kwargs, ...
               )
            else:
               ds = ds.map(
                   stage.fn, stage.fn_constructor_kwargs, ...
               )
        ds = ds.map(self.postprocess, ...)
        return ds

    def append_stage(self, name: str, stage: Stage):
        """Append a stage before postprocess.

        Args:
            name: The stage name.
            stage: The stage to append.
        """
        self.stages[name] = stage

    def list_stage_names(self) -> List[str]:
        """List the stage names of this processor in order.

        Returns:
            A list of stage names.
        """
        return list(self.stages.keys())

    def get_stage_by_name(self, name: str) -> Stage:
        """Get a particular stage by its name.

        Args:
            name: The stage name.

        Returns:
            The pipeline stage.
        """
        if name in self.stages:
            return self.stages[name]
        raise ValueError()
```

## Predefined Processor Configs

### VLLMProcessorConfig (SGLangProcessorConfig)

LLM processors can potentially have lots of variants based on different functionalities. For example, chat workloads would require applying a chat template to each input row; workloads with multi-modality inputs (e.g., images, audios, videos) would require a specialized preprocessor before sending to the LLM engine; models that could generate not only texts but images or videos would require a specialized postprocessor to transform output tokens. On the other hand, since the demand of some workloads is not high at this moment (e.g., video generation), we mainly cover text and image inputs with text only outputs for now, but our API is flexible enough to be extended to cover other workloads in the future.

```py
class VLLMProcessorConfig(ProcessorConfig):
    # The model name from HF, or the local/remote checkpoint path.
    model: str
    # The inference task type (generate, embed, etc). The default is "generate".
    task_type: str = "generate"
    # LLM engine specific kwargs. Can be auto-generated in the future.
    engine_kwargs: dict = {}
    # Additional Ray runtime environments.
    runtime_env: dict = {}

    # Processor configurations. Another option is to create different processor
    # configurations for different usages. For example, processor with or without
    # images can be separated.
    is_chat: bool = True
    has_images: bool = False
    need_tokenize: bool = True
    need_detokenize: bool = True

def build_vllm_processor(config, **kwargs) -> Processor:
    """Construct a Processor and configure stages."""

    processor = Processor(config, **kwargs)
    if config.is_chat:
        processor.append_stage(
            ApplyChatTemplate(
                fn_constructor_kwargs={...},
                map_kwargs={...},
            )
        )
    if config.has_images:
        processor.append_stage(
            ProcessImageStage(
                fn_constructor_kwargs={...},
                map_kwargs={...},
            )
        )
    if config.need_tokenize:
        processor.append_stage(
            TokenizeStage(
                fn_constructor_kwargs={...},
                map_kwargs={...},
            )
        )
    processor.append_stage(
        vLLMEngineStage(
            fn_constructor_kwargs={...},
            map_kwargs={...},
        )
    )
    if config.need_detokenize:
        processor.append_stage(
            DetokenizeStage(
                fn_constructor_kwargs={...},
                map_kwargs={...},
            )
        )
    return processor

PipelineBuilder.register(VLLMProcessorConfig, build_vllm_processor)
```

### HTTPRequestProcessorConfig

A processor that aims to query an endpoint with arbitrary functionalities. The endpoint can be LLMs, DB, storage, etc.

```py
class HTTPRequestProcessorConfig(ProcessorConfig):
    # The URL to query.
    url: str
    # The query header. Note that we will add
    # "Content-Type: application/json" to be the heahder for sure
    # because we only deal with requests body in JSON.
    header: Optional[str] = None
    # Queries per second. If None, the query will be sent sequentially.
    qps: Optional[int] = None

def build_http_request_processor(config, **kwargs) -> Processor:
    """Construct a Processor and configure stages."""
    processor = Processor(config, **kwargs)
    processor.append_stage(
        HTTPRequestStage(
            fn_constructor_kwargs={...},
            map_kwargs={...},
        )
    )
    return processor

PipelineBuilder.register(
    HTTPRequestProcessorConfig,
    build_http_request_processor
)
```

## Meeting Notes

Jan 7, 2025

Attendees: [Praveen Gorthy](mailto:praveeng@anyscale.com)[Cody Yu](mailto:cody@anyscale.com)[Hao Chen](mailto:chenh@anyscale.com)[Kourosh Hakhamaneshi](mailto:kourosh@anyscale.com)

Goals Of meeting

* Align on Batch inference API in this document

* Align on structure of code (at high level)  
  * Docs will live under a sub page in ray.data  
  * Implementation will be in \`ray.util.llm\` within OSS repo or “ray.llm” (like ray.dag today)  
      
* High level questions to answer?  
  * *Is LLMPipeline reimplementing parts of RayData or there parts of this pipeline that are better suited in RayData?*  
    * No. It is more like a user of Ray Data and simplifying a complex LLM pipeline for users.  
        
  * *LLMPipeline is a wrapper around Ray Data. Can it be an example like [this](https://huggingface.co/docs/datasets/en/quickstart)? How complex is LLMPipeline when compared to sequence of map batches?*  
    * Yes it is wrapper which is meant to reduce amount of information users need to run a typical batch llm pipeline  
    * Example can be very complex for users and its tedious for them to configure and understand how to get e2e working perfectly  
    * Also the current implementation in RayLLM-batch does few things like Image caching (leveraging ray data), sorting inputs for better prefix caching, gating requests before sending to vLLM to prevent vLLM from exploding.  
    * User Stories/Data points  
      * Instacart is [very interested in RayLLM-Batch like API](https://anyscale-external.slack.com/archives/C0298K92YBV/p1736275017509349)  
      * Nubank is [trying to do Batch inference](https://docs.google.com/document/d/1YLmTVPpGQ6SkWPaKhbTXTJzkG7mhtqruAr7h3R-PzMI/edit?tab=t.0) and realized they had to do a bunch of optimizations. Our API would make it much easier for them.  
      * Pinterest implemented Batchinference themselves using Ray Data. They had to [do lot of implementations](https://medium.com/pinterest-engineering/ray-batch-inference-at-pinterest-part-3-4faeb652e385) which come free with our API.  
  * *Who is the audience of this api?*  
    * Anyone who wants to scale LLM inference (who may or may not be experts of efficient usage of RayData, internals of using vLLM)

  * *Are we following a different approach than Ray LLM online? Where will code for that go? Is ray.llm.\_internal only for Batch LLM?*  
    * Ideally we follow a similar pattern for Online inference too. However scope of this api doc tab is for Batch inference.