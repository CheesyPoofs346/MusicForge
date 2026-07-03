"""Fine-tune ACE-Step with LoRA on training/dataset (made by prep.py).

Usage:
  python train.py                     # sane defaults for a 16GB GPU
  python train.py --max_steps 5000 --exp_name my_style_v2

LoRA adapters land in training/exps/logs/<exp_name>/<version>/checkpoints/.
Activate one for generation:  set ACE_LORA=<path to that folder>  then restart the server.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

args = [
    sys.executable, str(HERE / "trainer.py"),
    "--dataset_path", str(HERE / "dataset"),
    "--lora_config_path", str(HERE / "lora_config.json"),
    "--logger_dir", str(HERE / "exps" / "logs"),
    "--exp_name", "musicforge_lora",
    # ponytail: 16GB-friendly — bf16 weights, no dataloader workers (Windows),
    # checkpoint every 500 steps; bump max_steps for a real run
    "--precision", "bf16-true",
    "--num_workers", "0",
    "--max_steps", "1000",
    "--every_n_train_steps", "500",
    "--every_plot_step", "1000000",  # skip slow eval-generation during training
    *sys.argv[1:],  # your overrides win
]
print(" ".join(args[1:]))
sys.exit(subprocess.call(args, cwd=str(HERE)))
