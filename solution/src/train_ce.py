"""torchrun entry point: fine-tune the cross-encoder.
    torchrun --nproc_per_node=4 src/train_ce.py TOK_DIR PAIRS.npy OUT.pt [LR] [INIT.pt]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import crossenc  # noqa: E402

if __name__ == '__main__':
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-5
    init = sys.argv[5] if len(sys.argv) > 5 else None
    crossenc.train_ddp(sys.argv[1], sys.argv[2], sys.argv[3], lr=lr, init=init)
