"""All constants of the study in one place. Every number here goes into the config hash of the test log."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("P2_CACHE", Path.home() / ".cache" / "quant-research" / "p2"))
RAW = CACHE / "raw"
BUILT = CACHE / "built"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

USER_AGENT = "Mozilla/5.0 (research data client)"  # generic; no personal details in requests

# ---- sources (all free, no keys) -----------------------------------------------------------------------
GOYAL_PAGE = "https://sites.google.com/view/agoyal145"
GOYAL_URL = "https://drive.google.com/uc?export=download&id=17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1"
GOYAL_FILE = RAW / "goyal" / "PredictorData2025_gdrive.xlsx"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FRENCH_FILE = RAW / "french" / "F-F_Research_Data_Factors_CSV.zip"
JKP_MKT_URL = "https://data.jkpfactors.com/public/%5Ball_countries%5D_%5Bmkt%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip"
JKP_MKT_FILE = RAW / "jkp" / "all_countries_mkt_monthly_vw_cap.zip"
JKP_THEMES_URL = "https://data.jkpfactors.com/public/%5Ball_countries%5D_%5Ball_themes%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip"
JKP_THEMES_FILE = RAW / "jkp" / "all_countries_all_themes_monthly_vw_cap.zip"
PAPER_URL = "https://www.nber.org/system/files/working_papers/w30217/w30217.pdf"
NAGEL_URL = "https://www.nber.org/system/files/working_papers/w34104/w34104.pdf"

# ---- US predictors (KMZ footnote 32, Welch-Goyal mnemonics) ---------------------------------------------
US_PREDICTORS = ("dfy", "infl", "svar", "de", "lty", "tms", "tbl", "dfr", "dp", "dy", "ltr", "ep", "bm", "ntis", "lag_mkt")
INFL_EXTRA_LAG = 1  # CPI is released with a delay (Welch-Goyal convention)

# ---- JKP predictors (Part B, fixed in PLAN.md section 9) -------------------------------------------------
JKP_THEMES = ("accruals", "debt_issuance", "investment", "low_leverage", "low_risk", "momentum", "profit_growth",
              "profitability", "quality", "seasonality", "short_term_reversal", "size", "value")
JKP_PREDICTORS = JKP_THEMES + ("lag_mkt", "mkt_vol12")
JKP_MIN_STOCKS = 10
JKP_MIN_OOS = 120

# ---- the pre-registered specification (KMZ section 6) -------------------------------------------------------
T = 12  # rolling training window (months)
GAMMA = 2.0  # RFF bandwidth
P_GRID = (2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 36, 48, 60, 120, 240, 600, 1200, 3000, 6000, 12000)
Z_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
PRED_MIN_OBS = 36  # expanding-window predictor std needs 36 months
RET_VOL_WINDOW = 12  # returns are divided by their trailing 12-month std
BASE_SEED = 20240201
N_SEEDS_A = 300  # paper: 1,000 RFF draws; cut to 300 for the 2-core budget (PLAN.md changes log)
N_SEEDS_B = 30  # Part B, per country (PLAN.md changes log)
FEATURE_CHUNK = 600  # features per block when building window Gram matrices

# ---- periods (target months, i.e. the month whose return is forecast) -----------------------------------------
PAPER_FIRST = "1931-01"
PAPER_LAST = "2020-12"
POST_FIRST = "2021-01"
POST_JF_FIRST = "2024-01"  # after the JF publication (information only)

# ---- replication target (Table 1, Panel A, Nonlinear, c = 1000, z = 10^3) ----------------------------------------
TARGET_P = 12000
TARGET_Z = 1000.0
TARGET_SHARPE = 0.47
TARGET_TOL = 0.10
PAPER_TABLE1 = {"nonlinear_z1e3": {"R2": 0.006, "SR": 0.47, "SR_t": 4.5, "IR": 0.31, "IR_t": 2.9},
                "linear_z1e3": {"R2": -0.038, "SR": 0.46, "SR_t": 4.4, "IR": 0.33, "IR_t": 3.1},
                "linear_ridgeless": {"SR": -0.11, "SR_t": -1.0, "IR": -0.16, "IR_t": -1.6}}

# ---- reporting --------------------------------------------------------------------------------------------------
MAIN_Z = 1000.0
MAIN_P = (12, 120, 1200, 12000)  # c = 1, 10, 100, 1000
LINEAR_Z = (0.0,) + Z_GRID  # 0.0 means the ridgeless (minimum-norm) limit
SIMPLE_RULES = ("hist_mean", "tsmom_sign", "mom12", "kmz_mom", "nagel_volmom")
BOOT_REPS = 2000
NW_LAGS = 12
PERIODS_PER_YEAR = 12

# ---- return conventions ---------------------------------------------------------------------------------------
# "std": KMZ's pre-registered spec, returns divided by their trailing 12-month std (target, traded asset, benchmark).
# "raw": Nagel's (2025) convention, raw excess returns everywhere (secondary, added in the changes log).
CONVENTIONS = ("std", "raw")
