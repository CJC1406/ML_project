import logging
import random


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def seed_everything(seed: int) -> None:
    random.seed(seed)
