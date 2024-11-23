import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    try:
        _ = torch.zeros(1).to("mps")
        DEVICE = "mps"
    except:
        DEVICE = "cpu"
else:
    DEVICE = "cpu"


def reset():
    logger.info("Resetting devices...")
    logger.info(f"Using device: {DEVICE}")

    torch.manual_seed(SEED)

    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    elif DEVICE == "cpu":
        torch.backends.mkldnn.enabled = True
        torch.set_num_threads(4)
