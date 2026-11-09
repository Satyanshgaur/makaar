# Handwriting adapter (free tier)

`train_lora.py` runs as a private Kaggle script (T4, internet on, dataset `makaar-handwriting-eval` attached):

    kaggle kernels push -p <dir with train_lora.py and kernel-metadata.json> --accelerator NvidiaTeslaT4

It cuts printed replay blocks from DocLayNet and labels them with the base model on the second T4 while the first trains, checks the base model on 60 eval items, builds the mix (DATASETS.md), trains a rank-64 LoRA on the language model for `TRAIN_SECONDS`, scores base and adapter on the 450 eval items with makaar's metric and on 150 held-out printed blocks against the base model's own reading (forgetting check), and saves `adapter/`.

To use the adapter with llama.cpp:

    # PEFT names the layers model.language_model.layers.N; the converter expects model.layers.N
    python - <<'PY'
    from safetensors.torch import load_file, save_file
    t = load_file("adapter/adapter_model.safetensors")
    save_file({k.replace("model.language_model.layers.", "model.layers."): v.contiguous() for k, v in t.items()}, "renamed/adapter_model.safetensors")
    PY
    cp adapter/adapter_config.json renamed/
    python llama.cpp/convert_lora_to_gguf.py renamed --base models/paddleocr-vl-1.6-hf --outtype f16 --outfile handwriting-lora.gguf
    cp handwriting-lora.gguf models/paddleocr-vl-1.6/

The adapter is CC BY-NC-SA 4.0 (free, non-commercial use only); see DATASETS.md.
