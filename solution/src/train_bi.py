"""torchrun entry point: fine-tune the bi-encoder.
    torchrun --nproc_per_node=2 src/train_bi.py TOK_DIR PAIRS.npy OUT.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import biencoder  # noqa: E402

if __name__ == '__main__':
    biencoder.train_ddp(sys.argv[1], sys.argv[2], sys.argv[3])
