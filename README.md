Fork of AI2 https://github.com/allenai/OLMo

## Installation

You need PyTorch 2.1.2 to work nicely with MS-AMP

```bash
git clone https://github.com/thepowerfuldeez/OLMo.git
cd OLMo
pip install -e .[all]
pip install flash-attn==2.3.3 --no-build-isolation
```

## Research log

2024-04-01
----------
    
Tokenized dolma sample dataset (300B tokens). Downloaded from dolma dataset https://huggingface.co/datasets/allenai/dolma, took every 10th line, then pip install dolma and
    
`dolma tokens --documents "/mnt/harddrive/datasets/text/dolma/*.gz" --tokenizer.name_or_path "allenai/OLMo-7B" --tokenizer.eos_token_id 50279 --tokenizer.pad_token_id 1 --destination /mnt/harddrive/datasets/text/preprocessed/olmo-mix/v1_6_subset/allenai_gpt-neox-olmo-dolma-v1_5/ --processes 16`

Latest torch didn’t work due to bugs with ms-amp <-> torch 2.2.2. Installed torch 2.1.2 and flash-attn 2.3.3

RuntimeError: FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800
- Changed n_heads = 8 and d_model = 1536

Model is 300M param with embeddings. Benchmarked speed without flash attn and with flash attn. flash-attn from separate library. Got 16k tps without flash-attn and 17k with it. 6.2% improvement.

`git config --global --add safe.directory /hostroot/home/george/OLMo && export WANDB_API_KEY="" && torchrun --nproc_per_node=2 scripts/train.py configs/official/OLMo-250M.yaml --save_overwrite`

Strangely after updating to torch 2.3 as part of nvcr.io official docker image and using flash-attn 2 that was included, performance degraded back to 16100 tps.

Benchmarked torch.compile – 17900 tps

Increased batch from 4 to 6 – 23000 tps

Could'nt get fp8 up and running yet with MS-AMP, that's why switched to modern docker image.


### Training

The configs used to train the official OLMo models are provided in the [`configs/official/`](https://github.com/allenai/OLMo/blob/main/configs/official) directory.

Note that while the training and validation data is public and free to download, the paths to the data within those configs are pointed at a CloudFlare R2 bucket, which requires an API key for programmatic access.
So in order to use any of these configs to reproduce a training run you'll first have to download the corresponding data to a location of your choosing and then update the paths in the config accordingly.

You can derive the public HTTP URL from an R2 URL by replacing `r2://olmo-data` with `https://olmo-data.org`.
For example, if the R2 data URL is:

`r2://olmo-data/preprocessed/olmo-mix/v1_5/gpt-neox-20b-pii-special/part-000-00000.npy`

then the corresponding public URL is:

`https://olmo-data.org/preprocessed/olmo-mix/v1_5/gpt-neox-20b-pii-special/part-000-00000.npy`

Once you've updated the data paths in the config you can launch a training run via `torchrun`. For example, to launch the 1B model training on a single 8x GPU node, you would run:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/official/OLMo-1B.yaml
```

You can use the same method to launch multi-node jobs as well. See [the documentation](https://pytorch.org/docs/stable/elastic/run.html) for `torchrun` to understand the additional arguments you'll need to configure the rendezvous backend / endpoint.

To resume training from a checkpoint, you can pass its path (local or URL)
to `scripts/train.py` with the `--load_path` arguments. For example, to resume training from step 1000 of the OLMo 1B run:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/official/OLMo-1B.yaml --load_path https://olmo-checkpoints.org/ai2-llm/olmo-small/w1r5xfzt/step1000-unsharded
```

### Inspecting training data

You may be interesting in inspecting the exact tokens that composed a particular batch during the training of one of the OLMo models.
We provide tools to do this, but first you'll need to download the data as above (unless you have an R2 API key) and update the corresponding config accordingly.

Then take note of the URL of the data order file you want, which can be found in the [Models Overview](#models-overview) table. For example, the data order file for the first epoch of the OLMo-7B model is [https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy](https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy).

Once you have that you can use this snippet to inspect the data within a particular batch:

```python
import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

# Update these paths to what you want:
data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
train_config_path = "configs/official/OLMo-7B.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)


def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


# Get all 2048 x 2048 token IDs in the first batch.
get_batch_instances(0)
```


## Fine-tuning

To fine-tune an OLMo model using our trainer you'll first need to prepare your dataset by tokenizing it and saving the tokens IDs to a flat numpy memory-mapped array. See [`scripts/prepare_tulu_data.py`](./scripts/prepare_tulu_data.py) for an example with the Tulu V2 dataset, which can be easily modified for other datasets.

Next, prepare your training config. There are many examples in the [`configs/`](https://github.com/allenai/OLMo/blob/main/configs) directory that you can use as a starting point. The most important thing is to make sure the model parameters (the `model` field in the config) match up with the checkpoint you're starting from. To be safe you can always start from the config that comes with the model checkpoint. At a minimum you'll need to make the following changes to the config or provide the corresponding overrides from the command line:

- Update `load_path` to point to the checkpoint you want to start from.
- Set `reset_trainer_state` to `true`.
- Update `data.paths` to point to the `token_ids.npy` file you generated.
- Optionally update `data.label_mask_paths` to point to the `label_mask.npy` file you generated, unless you don't need special masking for the loss.
- Update `evaluators` to add/remove in-loop evaluations.

Once you're satisfied with your training config, you can launch the training job via `torchrun`. For example:

```
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config} \
    --data.paths=[{path_to_data}/input_ids.npy] \
    --data.label_mask_paths=[{path_to_data}/label_mask.npy] \
    --load_path={path_to_checkpoint} \
    --reset_trainer_state
```

Note: passing CLI overrides like `--reset_trainer_state` is only necessary if you didn't update those fields in your config.

## Evaluation

Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/ai2-olmo-eval) repo.

## Citing

```bibtex
@article{OLMo,
  title={OLMo: Accelerating the Science of Language Models},
  author={Dirk Groeneveld and Iz Beltagy and Pete Walsh and Akshita Bhagia and Rodney Kinney and Oyvind Tafjord and A. Jha and Hamish Ivison and Ian Magnusson and Yizhong Wang and Shane Arora and David Atkinson and Russell Authur and Khyathi Raghavi Chandu and Arman Cohan and Jennifer Dumas and Yanai Elazar and Yuling Gu and Jack Hessel and Tushar Khot and William Merrill and Jacob Daniel Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Valentina Pyatkin and Abhilasha Ravichander and Dustin Schwenk and Saurabh Shah and Will Smith and Emma Strubell and Nishant Subramani and Mitchell Wortsman and Pradeep Dasigi and Nathan Lambert and Kyle Richardson and Luke Zettlemoyer and Jesse Dodge and Kyle Lo and Luca Soldaini and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365485},
  journal={arXiv preprint},
}
```
