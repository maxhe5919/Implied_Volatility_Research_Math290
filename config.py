import numpy as np
import logging
DATA_DIR = "data/market_data"
SYMBOL = "NVDA"
DATE_STR = "2026-03-11"
RISK_FREE_RATE = 0.0421

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MONEYNESS_NODES = np.linspace(0.80, 1.20, 21)
MAX_DTE_TERMS = 5
ROLLING_WINDOW = 60