# Overview

The majority of post-training involves spending time on evals, as well as data generation, and curation. Effectively the same pipeline that is used for scaling evaluations can be used to scale up data generation as a workload. So it’s very crucial to have a library that makes this seamless and easy at any arbitrary scale, hence why we need this library. 

# Key features

## Scalable execution

* Support for both single-node and multi-node execution  
* Compatible with any homogeneous or heterogeneous machine configuration.  
* Horizontal scaling capabilities for data generation and evaluation

## Modular and reusable components

* Support for both API-based models (e.g., OpenAI) and local models via vLLM  
* Reusable components for evaluation, data synthesis, and reinforcement learning

## Standardized set of evaluations 

* Library of standard configurable reasoning benchmarks in math and coding  
* CLI based interface for quick on demand eval runs

# Quick start

```py
from skythought.evals import SkyEnvBuilder
# Build environment for specific benchmark
env = SkyEnvBuilder.build("AIME24")

# Score can be used standalone in RL pipelines
env.score({"generation": "...", "expected": "..."})

# Read and optionally limit dataset
ds = env.read_dataset()
ds = ds.limit(10)  # Optional: Test on smaller scale

# Setup environment with model and scaling configurations
env.setup(model_config, scaling_config, template)

# Run generation pipeline
ds = env.generate(ds)
ds = env.score(ds)

# Save results
env.save(ds, "./results/aime24_llama/", format="json")
```

Results will be stored in:

```
./results/aime24_llama/
├── generations/
├── scores/
└── logs.txt 
```

## Component details

### Env setup

You can reuse the same env with different models, different prompt templates, different sampling parameters, different deployment and scaling options. For this you need to set up your env.

```py
# Build environment for specific benchmark
env = SkyEnvBuilder.build("AIME24")

# Configure model and scaling parameters
env.setup(
    model_config=model_config,
    scaling_config=scaling_config,
    template=template,
    sampling_params={"n": 1024, "temperature": 0.7}
)

```

 

### Model configuration and scaling config

It supports both scaling up client code for hitting a scalable hosted server (e.g. openAI, Anthropic, etc) or local deployments via vLLM.

```py
# HTTP 
env.setup(
model_config=HTTPConfig(model=..., api_url=..., api_key=...),
scaling_config=ScalingConfig(concurrency=2),
)

# VLLM
env.setup(
model_config=VLLMConfig(model=..., engine_kwargs=...),
scaling_config=ScalingConfig(concurrency=2, accelerator_type="L40S")
)
```

### Prompt templates

Flexible templating for chat and non-chat formats. Each task also comes with its default template that can be reused or changed as needed.

```
# non-chat
template: "<Question>{question}</Question><answer>"

# Chat
template:
	- {"role": "system", "content": "You should think step by step"}
	- {"role": "user", "content": "Answer the following question: {question}"}

env.setup(..., template=template)
```

### Reusable components 

```py
env.setup(..., template={...})

# point-wise operations
env.apply_prompt_template_on_item({"question": ...})
env.score_item({"question": ..., "expected": ...})

# dataset operation
ds = env.generate(ds)
ds = env.score(ds)
```

### Pre-configured benchmark environments

Each env can have some default setting (such as dataset, split, prompt, etc), but also can be configured through the allowed knobs for better customization

```py
env = SkyEnvBuilder.build("OlympiadBench", {"score_format": False, "subset": "easy"}
```

### Fault-tolerance baked-in

If there are certain records that timeout / error out the records are skipped and deferred to later analysis. 

```py
# Run generation pipeline
ds = env.generate(ds)
ds = env.score(ds)

# Save results
env.save(ds, "./results/aime24_llama/", format="json")

# Assuming some records timed out
./results/aime24_llama/
├── generations/
├── scores/
├── failures/
└── logs.txt 
```

### Lazy imports

Env utilities are imported lazily. If the env requires other dependencies you’ll only get error with you build and not when you import. For example let’s say `MATH500` requires `latex2sympy2` and the system does not have it:

```py
from skythought.evals import SkyEnvBuilder # Should run without any errors

env = SkyEnvBuilder.build("MATH500") # raise an error of missing latex2sympy2

env = SkyEnvBuilder.build("GSM8K") # runs fine
```

### CLI 

The library comes with a high-level CLI that you can invoke to run standard benchmarks for reasoning. It works with yamls with the option to override them (uses [hydra](https://github.com/facebookresearch/hydra) under the hood). 

```
# AIME24.yaml
env:
  name: AIME24
  env_kwargs:
    metrics:
      - {"type": "pass", "k": 1, "n_samples": 64}

model_config:
  type: vllm
  model_source: Qwen/Qwen2-7B-Instruct
  engine_kwargs:
    tensor_parallel_size: 4
  concurrency: 8

template:
  - {"role": "system", "content": "You should think step by step"}
  - {"role": "user", "content": "Answer the following question: {question}"}

sampling_params:
  max_tokens: 4096
  temperature: 0.5


logging:
  experiment_name: {$env.name}_{$model_config.model_source}_{$sampling_params.temperature}
  results_dir: ./{$SKYHOME}/evaluation_results/{$logging.experiment_name}
```

CLI call will be something like: 

```py
python -m skythought.evals.cli evaluate AIME.yaml --model_config.model_source Novasky/Sky-T1
```

## 

## Case studies

### **Rejection sampling data generation on Math**

We want to subsample NUMINA and want to do generation on QWQ and only keep the ones that are readable and have correct answers. 

```py
# Define the same math functionality with Numina dataset
class NuminaMathEnv(MathEnv):
	...
	def scoring_functions():
		return {"readability": ReadabilityScorer, "math": MathScorer}

SkyEnvBuilder.register("NUMINA", NuminaMathEnv)

# Builds the env and limit the dataset
env = SkyEnvBuilder.build("NUMINA")
ds = env.read_dataset()
ds = ds.limit(5000)

# Setup the env with the desired model and scaling parameters
env.setup(...)

# Generate and score with math 
ds = env.generate(ds)
ds = env.score(ds)

Readabi

# filter ds
ds = ds.filter(lambda row: row["math"] > 0 and row["readabilty"] > 0)

env.save(ds, "./results/numina-filtered/")
```

We then might want to re-write them using another model to a more readable format for supervised training. For this we can use `ray.data.llm` to run the batched op on top of this:

```py
from ray.data.llm import build_llm_processor, VLLMProcessorConfig
config =  VLLMProcessorConfig(
    model="meta-llama/Llama-3.1-70B-Instruct",
    engine_kwargs=dict(pipeline_parallel_size=2),
)
ds = build_processor(
	config,
	preprocessor_fn=lambda row: dict(
       	messages=[
            		{"role": "system", "content": ...},
           	{"role": "system", "content": ... row["question"] ...},
        	],
       sampling_params=dict(...),
),
postprocess_fn=lambda row: dict(rewritten_answer=["generated_text"]),
accelerator_type="L40S",
concurrency=2
)(ds)
```

### **Best of N evaluation**

In here, we want to sample n generations per data point in the data and then do a grouped average score across each question group.

```py

# Build and setup env
env = ...
env.setup(..., sampling_params={"n": 1024})

# Read default ds and attach generator and scorer to it
ds = env.read_dataset()
ds = env.generate(ds)
ds = env.score(ds)

# Define a group aggregator (average across group of questions)
def group_map_fn(group):
    
    return {
        "question": [group["question"].iloc[0]],
        "question_hash": [group["question_hash"].iloc[0]],
        "generated_texts": [list(group["generated_text"])],
        "math_scores": [list(group["math_score"])],
        "math_score_avg": [np.mean(list(group["math_score"]))],
        "math_score_max": [np.max(list(group["math_score"]))],
    }
    
ds_grouped = scoring_ds.groupby(["question", "question_hash"]).map_groups(group_map_fn)

best_of_n_score = ds.mean(on="math_score_max")
```

# Implementation details

The main class is Env.

```py
class Env:
	
	def setup(self, ...):
		pass
	
	# Dataset operators
	def read_dataset(self):
		pass

	def generate(self, dataset):
		pass

	def score(self, dataset):
		pass

	# helpers for items
	def score_item(self, item: Dict[str, Any]):
		pass

	def apply_prompt_template_on_item(self, item: Dict[str, Any]):
		pass
	
	# IO
	def save(self, path, format):
		pass
```

