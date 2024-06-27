# 🇮🇹 Loquace 🇮🇹 
# An exclusively Italian speaking, instruction finetuned, Large Language model. 🇮🇹

The Loquace Italian LLM models are created as a proof-of-concept to evaluate on how language tuning can be achieved using QLoRa by instruct-tunings foundational LLMs using dataset of a specific language.

The QLoRa (https://github.com/artidoro/qlora) method of fine-tuning significantly lower the resources requirements compared to any other methods available, this allow to easily execute the process on significanly larger dataset while still using consumers GPUs and still achieve high accuracy.

You can find the big Loquace family on HuggingFace:

# LATEST MODEL!!!!
https://huggingface.co/cosimoiaia/Loquace-7B-Mistral -  Based on Mistral-7B-Instruct

### OLD MODELS:
https://huggingface.co/cosimoiaia/Loquace-70m   -   Based on pythia-70m

https://huggingface.co/cosimoiaia/Loquace-410m  -   Based on pythia-410m

https://huggingface.co/cosimoiaia/Loquace-7B    -   Based on Falcon-7B.

https://huggingface.co/cosimoiaia/Loquace-12B   -   Based on pythia-12B

https://huggingface.co/cosimoiaia/Loquace-20B   -   Based on gpt-neox-20B

## Project Installation

### Prerequisites

Make sure you have CUDA installed if you have an NVIDIA GPU. You can download CUDA from the NVIDIA website: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### CUDA Configuration (Windows Only)

1. Find the CUDA installation directory. It is usually located at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`, where `X.X` represents the installed version of CUDA.
2. Open CMD ad an Administrator and set the `CUDA_HOME` environment variable:
   ```
   setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X" /m
   ```

3. Add CUDA to the `PATH`:
   Open PowerShell as an Administrator and run the following command:
   ```
   [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\libnvvp", [System.EnvironmentVariableTarget]::Machine)

   ```

### Install the dependencies

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U -r requirements.txt
```
Next, install unsloth, I created a fork that is windows compatible:
```
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/LucaCatone/unsloth-windows.git"
```
Or if you are on linux
```
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
```


## 🏋️ Reproduce the training 
To replicate the results using the Loquace dataset:

```
python3 qlora.py \
    --model_name_or_path model_path \
    --output_dir ./Loquace-XX \
    --dataset loquace \
    --do_train True \
    --do_eval True \
    --do_mmlu_eval False \
    --source_max_len 512 \
    --target_max_len 512 \
    --logging_steps 100 \
    --max_steps 10000 \
    --save_strategy steps \
    --data_seed 69420 \
    --save_steps 5000 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --eval_steps 1000 \
    --optim paged_adamw_32bit
```

Alternatively you can use the Dockerfile included.

Once the training is done, you can merge the checkpoints with the original model using the `merge.py` script:
```
python3 merge.py --base_model_name_or_path base_model --peft_model_path checkpoint-XXXX/adapter_model/ --output_dir final_model/ 
```


Special thanks to Genesis Cloud for kindly providing the infrastructure and the GPU Computing. (https://gnsiscld.co/26qhlf)
