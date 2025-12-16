# CLAUDE.md

## Overview

This is **keiba-yosou** (競馬予想AI) - a collaborative human-AI horse racing prediction system for Japanese JRA races. The system combines machine learning base predictions with human scenario analysis (pace, track bias, racing conditions) to produce calibrated probability estimates for betting strategy optimization.

**Last Updated**: 2025-12-16
**Project Status**: Active development, Phase 1 complete
**Primary Language**: Python 3.12
**Target Domain**: JRA (Japan Racing Association) horse racing

---

## Architecture Overview

### Three-Layer System

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: UI / API                                  │
│  - Next.js 14 UI (keiba-ui/)                        │
│  - React+Vite UI (scenario-ui/)                     │
│  - FastAPI backend (src/api.py)                     │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Scenario Adjustment                       │
│  - Human scenario inputs (pace, bias, conditions)   │
│  - Log-odds space adjustments                       │
│  - Probability integration & normalization          │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Layer 1: Base Prediction                           │
│  - LightGBM models (win/in3 targets)                │
│  - Feature engineering (30+ features)               │
│  - SQLite feature store                             │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Layer 0: Data Ingestion                            │
│  - netkeiba.com scraping (premium member)           │
│  - SQLite database (netkeiba.db)                    │
│  - Race results, lap times, horse history           │
└─────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
keiba-yosou/
├── src/                          # Core Python backend
│   ├── ingestion/                # Web scraping & data collection
│   │   ├── scraper.py            # HTTP client with Cookie auth
│   │   ├── parser.py             # HTML parsing (BeautifulSoup)
│   │   ├── parser_race_result.py # Race result parsing
│   │   ├── parser_horse_past.py  # Horse history parsing
│   │   ├── parser_horse_basic.py # Horse basic info
│   │   ├── models.py             # Dataclass definitions
│   │   ├── db.py                 # SQLite operations
│   │   ├── ingest_runner.py      # CLI entry point
│   │   ├── race_list.py          # Race ID fetching
│   │   └── fetcher.py            # High-level orchestration
│   │
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py           # build_feature_table entry
│   │   ├── feature_builder.py    # Main feature construction (v3.3.1)
│   │   └── sqlite_store_feature.py # Feature table DDL
│   │
│   ├── models/                   # Machine learning models
│   │   ├── train_lgbm.py         # LightGBM training (v2.0)
│   │   ├── predict_lgbm.py       # Prediction runner
│   │   └── model_utils.py        # Data loading, splits
│   │
│   ├── scenario/                 # Scenario adjustment system
│   │   ├── spec.py               # ScenarioSpec dataclass
│   │   ├── types.py              # Type definitions
│   │   ├── adjuster.py           # Adjustment logic
│   │   ├── runner.py             # Scenario execution
│   │   ├── score.py              # Output format
│   │   ├── compare.py            # Multi-scenario comparison
│   │   └── ui.py                 # UI integration
│   │
│   ├── calibration.py            # Probability calibration (v3)
│   ├── baba_adjustment.py        # Track condition adjustment (v2.0)
│   ├── pace_adjustment.py        # Pace-based adjustment (v2.0)
│   ├── pace_prediction.py        # Lap time prediction
│   ├── probability_integration.py # Combine all adjustments
│   ├── backtest.py               # Strategy backtesting
│   ├── shap_explainer.py         # SHAP explanations
│   ├── api.py                    # FastAPI endpoints
│   └── timeline_manager.py       # Time-series split, leak prevention
│
├── keiba-ui/                     # Next.js 14 UI (App Router)
│   ├── app/
│   │   ├── scenario/page.tsx     # Main scenario page
│   │   ├── components/           # React components
│   │   └── globals.css           # Tailwind + custom CSS
│   ├── public/                   # Static scenario JSONs
│   └── package.json
│
├── scenario-ui/                  # React+Vite alternative UI
│   ├── src/
│   │   ├── ScenarioApp.tsx
│   │   └── components/
│   └── package.json
│
├── scripts/                      # CLI scripts
│   ├── train_eval_logistic.py    # Train models
│   ├── run_scenario_prediction.py # Run predictions
│   ├── run_compare_scenarios.py  # Compare scenarios
│   ├── run_scenario_ui.py        # Launch UI server
│   └── horse_lap_coverage.py     # Data coverage analysis
│
├── models/                       # Trained model artifacts
│   ├── lgbm_target_win.txt
│   ├── lgbm_target_in3.txt
│   ├── feature_columns_*.json
│   └── base_*_model_*_meta.json
│
├── features/                     # Feature definitions
├── out/                          # Output files
├── data/                         # Local databases (gitignored)
│
├── requirements.txt              # Python dependencies
├── README.md                     # Japanese documentation
├── AGENT.md                      # Development agent guidelines
├── GIT_SETUP.md                  # Git workflow guide
└── CLAUDE.md                     # This file
```

---

## Database Schema

### Primary Database: `netkeiba.db` (SQLite)

**Location**: `src/netkeiba.db` (gitignored - generated locally)

#### Core Tables

**races** - Race basic information
```sql
CREATE TABLE races (
    race_id TEXT PRIMARY KEY,        -- 12-digit ID (YYYYJJKKNNRR)
    date TEXT,                        -- YYYY-MM-DD
    place TEXT,                       -- 東京, 中山, 京都, etc.
    kai INTEGER,                      -- Meeting number
    nichime INTEGER,                  -- Day number
    race_no INTEGER,                  -- Race number
    name TEXT,                        -- Race name
    grade TEXT,                       -- G1, G2, G3, OP, Listed
    race_class TEXT,                  -- Class level
    course_type TEXT,                 -- turf, dirt, steeple
    distance INTEGER,                 -- Meters
    course_turn TEXT,                 -- right, left, straight
    course_inout TEXT,                -- inner, outer
    weather TEXT,                     -- 晴, 曇, 雨, etc.
    track_condition TEXT,             -- 良, 稍重, 重, 不良
    start_time TEXT,
    baba_index INTEGER,
    baba_comment TEXT,
    analysis_comment TEXT,
    head_count INTEGER
);
```

**race_results** - Horse performance in each race
```sql
CREATE TABLE race_results (
    race_id TEXT,
    horse_id TEXT,
    finish_order INTEGER,             -- 1, 2, 3, ... or NULL (DNS/DNF)
    finish_status TEXT,               -- 正常, 取消, 除外, etc.
    frame_no INTEGER,                 -- 枠番 (1-8)
    horse_no INTEGER,                 -- 馬番 (1-18)
    horse_name TEXT,
    sex TEXT,                         -- 牡, 牝, セ
    age INTEGER,
    weight REAL,                      -- 斤量 (kg)
    jockey_id TEXT,
    jockey_name TEXT,
    time_str TEXT,                    -- "1:23.4"
    time_sec REAL,                    -- Seconds
    margin TEXT,                      -- Margin string
    passing_order TEXT,               -- "1-1-1-1"
    last_3f REAL,                     -- Last 600m time
    win_odds REAL,
    popularity INTEGER,               -- Betting popularity rank
    body_weight INTEGER,              -- 馬体重 (kg)
    body_weight_diff INTEGER,         -- Change from last run
    time_index INTEGER,
    trainer_id TEXT,
    trainer_name TEXT,
    trainer_region TEXT,
    owner_id TEXT,
    owner_name TEXT,
    prize_money INTEGER,
    remark_text TEXT,
    PRIMARY KEY (race_id, horse_id)
);
```

**payouts** - Betting payouts
```sql
CREATE TABLE payouts (
    race_id TEXT,
    bet_type TEXT,                    -- 単勝, 複勝, 馬連, etc.
    combination TEXT,                 -- "1", "1-2-3", etc.
    payout INTEGER,                   -- Yen (per 100 yen bet)
    popularity INTEGER,
    PRIMARY KEY (race_id, bet_type, combination)
);
```

**lap_times** - Race-wide lap times
```sql
CREATE TABLE lap_times (
    race_id TEXT,
    lap_index INTEGER,                -- 1, 2, 3, 4
    distance_m INTEGER,               -- 200, 400, 600, 800
    time_sec REAL,                    -- Lap time in seconds
    PRIMARY KEY (race_id, lap_index)
);
```

**horse_laps** - Individual horse lap times (Premium feature)
```sql
CREATE TABLE horse_laps (
    race_id TEXT,
    horse_id TEXT,
    section_m INTEGER,                -- 200, 400, 600... or 100, 300, 500...
    time_sec REAL,                    -- Section time in seconds
    position INTEGER,                 -- Position at that section (nullable)
    PRIMARY KEY (race_id, horse_id, section_m)
);
-- Note: Even distances use 200m increments, odd use 100m start + 200m
```

**corners** - Corner passing positions
```sql
CREATE TABLE corners (
    race_id TEXT PRIMARY KEY,
    corner_1 TEXT,                    -- "1,2,3,4,..."
    corner_2 TEXT,
    corner_3 TEXT,
    corner_4 TEXT
);
```

**short_comments** - Expert comments (Premium)
```sql
CREATE TABLE short_comments (
    race_id TEXT,
    horse_id TEXT,
    horse_name TEXT,
    finish_order INTEGER,
    comment TEXT,
    PRIMARY KEY (race_id, horse_id)
);
```

**feature_table** - ML training features (generated)
```sql
CREATE TABLE feature_table (
    -- Identity
    race_id TEXT,
    horse_id TEXT,

    -- Targets
    target_win INTEGER,               -- 1 if finished 1st, else 0
    target_in3 INTEGER,               -- 1 if finished in top 3, else 0
    target_value REAL,                -- Finish position value

    -- Race attributes
    course TEXT,                      -- Full course name
    surface TEXT,                     -- turf/dirt
    surface_id INTEGER,               -- Encoded surface
    distance INTEGER,
    distance_cat TEXT,                -- short/mile/middle/long
    track_condition TEXT,
    track_condition_id INTEGER,
    field_size INTEGER,               -- Number of horses
    race_class TEXT,
    race_year INTEGER,
    race_month INTEGER,

    -- Horse attributes
    waku INTEGER,                     -- Frame number
    umaban INTEGER,                   -- Horse number
    horse_weight REAL,
    horse_weight_diff REAL,
    is_first_run INTEGER,             -- Boolean

    -- Historical stats - Overall
    n_starts_total INTEGER,
    win_rate_total REAL,
    in3_rate_total REAL,
    avg_finish_total REAL,
    std_finish_total REAL,

    -- Historical stats - Distance category
    n_starts_dist_cat INTEGER,
    win_rate_dist_cat REAL,
    in3_rate_dist_cat REAL,
    avg_finish_dist_cat REAL,
    avg_last3f_dist_cat REAL,

    -- Recent form
    days_since_last_run REAL,
    recent_avg_finish_3 REAL,         -- Avg finish in last 3 runs
    recent_best_finish_3 INTEGER,
    recent_avg_last3f_3 REAL,

    -- Condition-specific stats
    n_starts_track_condition INTEGER,
    win_rate_track_condition REAL,

    -- Course-specific stats
    n_starts_course INTEGER,
    win_rate_course REAL,

    -- Weight
    avg_horse_weight REAL,

    -- Lap ratio features (relative to race average)
    hlap_overall_vs_race REAL,        -- Overall pace vs race avg
    hlap_early_vs_race REAL,          -- 0-40% distance zone
    hlap_mid_vs_race REAL,            -- 40-80% distance zone
    hlap_late_vs_race REAL,           -- 80-100% distance zone
    hlap_last600_vs_race REAL,        -- Final 600m vs race avg

    PRIMARY KEY (race_id, horse_id)
);
```

---

## Data Ingestion

### Prerequisites

1. **netkeiba Premium Membership** - Required for:
   - Individual horse lap times (`horse_laps`)
   - Expert short comments
   - Full race details

2. **Cookie Configuration**
   ```bash
   cp src/ingestion/.env.example src/.env
   # Edit .env with your netkeiba cookies
   ```

   Required cookies:
   - `NETKEIBA_COOKIE_netkeiba`
   - `NETKEIBA_COOKIE_nkauth`
   - `NETKEIBA_COOKIE_ga_netkeiba_member`
   - Additional cookies from browser DevTools

### Scraper Architecture

**Key Features**:
- 12 User-Agent rotation for bot detection avoidance
- Exponential backoff retry for 500 errors (up to 4 retries)
- User-Agent retry for 400 Bad Request
- Random sleep 2-3.5 seconds between requests
- Cookie-based authentication
- Idempotent design (safe to re-run)

**Files**:
- `src/ingestion/scraper.py` - HTTP client with retry logic
- `src/ingestion/parser.py` - BeautifulSoup HTML parsing
- `src/ingestion/db.py` - SQLite upsert operations
- `src/ingestion/ingest_runner.py` - CLI orchestration

### Running Ingestion

```bash
# Full year ingestion
cd src
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024

# Specific races
python -m ingestion.ingest_runner --race-ids 202406050811 202408070108 -v

# Specific venue
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024 --jyo 05

# Dry run (no DB writes)
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024 --dry-run

# Skip existing races
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024 --skip-existing
```

**Venue Codes** (`--jyo`):
- 01: 札幌 (Sapporo)
- 02: 函館 (Hakodate)
- 03: 福島 (Fukushima)
- 04: 新潟 (Niigata)
- 05: 東京 (Tokyo)
- 06: 中山 (Nakayama)
- 07: 中京 (Chukyo)
- 08: 京都 (Kyoto)
- 09: 阪神 (Hanshin)
- 10: 小倉 (Kokura)

### Race ID Format

12-digit format: `YYYYJJKKNNRR`
- `YYYY`: Year (2024)
- `JJ`: Venue code (01-10)
- `KK`: Meeting number (01-99)
- `NN`: Day number (01-99)
- `RR`: Race number (01-12)

Example: `202406050811` = 2024, Tokyo (06), 5th meeting, 8th day, 11th race (Arima Kinen)

---

## Feature Engineering

### Feature Builder Architecture

**File**: `src/features/feature_builder.py` (v3.3.1)

**Key Design Principles**:
1. **No Future Information Leakage** - Strict datetime filtering for past runs
2. **Schema Flexibility** - Handles missing columns gracefully
3. **Race Attributes Caching** - Performance optimization
4. **Lap Ratio Features** - Individual vs race average comparisons

### Feature Categories

#### 1. Global Historical Stats
- `n_starts_total`, `win_rate_total`, `in3_rate_total`
- `avg_finish_total`, `std_finish_total`

#### 2. Distance Category Stats
- `n_starts_dist_cat`, `win_rate_dist_cat`, `in3_rate_dist_cat`
- `avg_finish_dist_cat`, `avg_last3f_dist_cat`

Distance categories:
- **short**: < 1400m
- **mile**: 1400-1800m
- **middle**: 1800-2200m
- **long**: >= 2200m

#### 3. Recent Form (Last 3 Runs)
- `days_since_last_run`
- `recent_avg_finish_3`, `recent_best_finish_3`
- `recent_avg_last3f_3`

#### 4. Condition-Specific Stats
- `n_starts_track_condition`, `win_rate_track_condition`
- Track conditions: 良 (firm), 稍重 (good), 重 (yielding), 不良 (soft)

#### 5. Course-Specific Stats
- `n_starts_course`, `win_rate_course`

#### 6. Weight Features
- `avg_horse_weight` - Historical average
- `horse_weight` - Current weight
- `horse_weight_diff` - Change from last run

#### 7. Lap Ratio Features (hlap_*)

**Premium feature requiring individual horse lap times**

```python
hlap_overall_vs_race    # Overall pace vs race average
hlap_early_vs_race      # 0-40% distance zone
hlap_mid_vs_race        # 40-80% distance zone
hlap_late_vs_race       # 80-100% distance zone
hlap_last600_vs_race    # Final 600m vs race average
```

**Calculation**:
1. For each section: `delta = horse_lap_time - race_avg_lap_time`
2. For each zone: Average all `delta` values in that zone
3. Negative = faster than average, Positive = slower than average

**Important**: These features are computed in `compute_lap_ratio_features_for_race()` and require both `horse_laps` and `lap_times` tables populated.

### Building Feature Table

```bash
# Method 1: Via Python
cd src
python -c "
import sqlite3
import logging
logging.basicConfig(level=logging.INFO)
from features import build_feature_table
conn = sqlite3.connect('netkeiba.db')
build_feature_table(conn)
"

# Method 2: Via script
python run_build_features.py

# Rebuild from scratch
python -c "
import sqlite3
conn = sqlite3.connect('netkeiba.db')
conn.execute('DROP TABLE IF EXISTS feature_table')
conn.commit()
conn.close()
"
python run_build_features.py
```

### Verifying Features

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('src/netkeiba.db')

# Check row count
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM feature_table")
print(f"Total rows: {cursor.fetchone()[0]}")

# Check lap features
df = pd.read_sql_query("""
    SELECT
        race_id, horse_id,
        hlap_overall_vs_race,
        hlap_early_vs_race,
        hlap_mid_vs_race,
        hlap_late_vs_race,
        hlap_last600_vs_race
    FROM feature_table
    WHERE hlap_overall_vs_race IS NOT NULL
    LIMIT 20
""", conn)
print(df)

# Check for nulls
df_nulls = pd.read_sql_query("""
    SELECT
        SUM(CASE WHEN hlap_overall_vs_race IS NULL THEN 1 ELSE 0 END) as null_count,
        COUNT(*) as total_count
    FROM feature_table
""", conn)
print(df_nulls)
```

---

## Model Training

### LightGBM Models

**File**: `src/models/train_lgbm.py` (v2.0)

**Training Targets**:
1. `target_win` - Binary classification (1st place)
2. `target_in3` - Binary classification (top 3 finish)

**Model Configuration**:
```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'lambda_l2': 0.1,           # L2 regularization
    'min_child_samples': 20,
    'max_depth': 6,
}

num_boost_round = 1000
early_stopping_rounds = 50
```

### Training Workflow

```bash
# Train both models
cd src
python -m models.train_lgbm --db netkeiba.db --out ../models/

# Custom parameters
python -m models.train_lgbm \
    --db netkeiba.db \
    --out ../models/ \
    --test-split 0.2 \
    --val-split 0.15
```

**Outputs** (in `models/` directory):
- `lgbm_target_win.txt` - LightGBM model (text format)
- `lgbm_target_win.pkl` - Joblib serialized model
- `feature_columns_target_win.json` - Feature names
- `feature_importance_target_win.csv` - Feature importance
- Same files for `target_in3`

**Evaluation Metrics**:
- AUC (Area Under ROC Curve)
- Log Loss
- Accuracy
- Precision
- Recall
- F1 Score

### Feature Importance Analysis

After training, check feature importance:

```python
import pandas as pd

# Load feature importance
df_imp = pd.read_csv('models/feature_importance_target_win.csv')
df_imp = df_imp.sort_values('importance', ascending=False)
print(df_imp.head(20))
```

Common top features:
- `win_rate_dist_cat`
- `avg_last3f_dist_cat`
- `recent_avg_finish_3`
- `hlap_late_vs_race`
- `win_rate_course`
- `popularity` (if included)

---

## Probability Calibration & Adjustments

### 1. Calibration (src/calibration.py v3)

**Purpose**: Convert raw model probabilities to true probabilities

**Methods**:
- **Platt Scaling** - Logistic regression on model outputs
- **Isotonic Regression** - Non-parametric monotonic function

```python
from src.calibration import PlattCalibrator, IsotonicCalibrator

# Train calibrator
calibrator = PlattCalibrator()
calibrator.fit(y_true, y_pred_proba)

# Apply calibration
calibrated_probs = calibrator.predict(y_pred_proba)
```

### 2. Track Condition Adjustment (src/baba_adjustment.py v2.0)

**Purpose**: Adjust for track condition effects on horse performance

**Approach**: Log-odds space adjustment
```python
# For each horse, estimate P(win | track_condition)
# Convert to log-odds, apply delta, convert back
delta_logodds = learned_adjustment_for_track_condition
adjusted_prob = logistic(logit(base_prob) + delta_logodds)
```

### 3. Pace Adjustment (src/pace_adjustment.py v2.0)

**Purpose**: Adjust for race pace effects on running styles

**Running Styles** (脚質):
- 逃げ (Nige) - Front Runner
- 先行 (Senkou) - Stalker
- 差し (Sashi) - Closer
- 追込 (Oikomi) - Late Closer

**Pace Types**:
- S (Slow) - Benefits closers
- M (Middle) - Balanced
- H (High) - Benefits front runners

**Adjustment Logic**:
```python
# Predict front/back lap times
pace_balance = predict_pace_balance(race_features)

# Adjust by running style
if pace == "Slow" and style == "Closer":
    delta_logodds = +0.5  # Favorable
elif pace == "High" and style == "Front Runner":
    delta_logodds = +0.3  # Favorable
else:
    delta_logodds = calculate_adjustment(pace, style)
```

### 4. Probability Integration (src/probability_integration.py)

**Purpose**: Combine all adjustments and normalize

**Process**:
1. Start with calibrated base probability
2. Convert to log-odds space
3. Apply all adjustments (track, pace, etc.)
4. Convert back to probability
5. Apply Softmax normalization across all horses in race

```python
# Pseudo-code
log_odds = logit(calibrated_prob)
log_odds += delta_track_condition
log_odds += delta_pace
log_odds += delta_scenario  # From scenario module
final_prob = sigmoid(log_odds)

# Normalize across race
probs = softmax([final_prob_1, final_prob_2, ..., final_prob_n])
```

---

## Scenario System

### Purpose

Enable human experts to inject domain knowledge about:
- Expected race pace
- Track bias (inner/outer advantage)
- Track condition nuances
- Running style matchups

### Scenario Specification

**File**: `src/scenario/spec.py`

```python
@dataclass
class ScenarioSpec:
    # Pace prediction
    pace_type: PaceType              # S, M, H

    # Track bias
    bias_type: BiasType              # Inner, Outer, Flat
    bias_strength: float             # 0.0 to 1.0

    # Track condition
    track_condition: str             # 良, 稍重, 重, 不良
    moisture_turf_goal: Optional[float]
    moisture_turf_corner4: Optional[float]
    moisture_dirt_goal: Optional[float]
    moisture_dirt_corner4: Optional[float]
    cushion_value: Optional[float]

    # Weather
    weather: Optional[str]           # 晴, 曇, 雨

    # Comments
    scenario_name: str
    scenario_description: str
```

### Running Scenarios

```bash
# Single scenario prediction
python scripts/run_scenario_prediction.py \
    --race-id 202406050811 \
    --scenario-file public/scenario_slow_inner.json

# Compare multiple scenarios
python scripts/run_compare_scenarios.py \
    --race-id 202406050811 \
    --scenarios scenario_1.json scenario_2.json scenario_3.json
```

### Scenario JSON Format

```json
{
  "scenario_name": "Slow Pace + Inner Bias",
  "scenario_description": "Heavy rain, soft track favoring inside",
  "pace_type": "S",
  "bias_type": "Inner",
  "bias_strength": 0.7,
  "track_condition": "重",
  "moisture_turf_goal": 18.5,
  "moisture_turf_corner4": 19.2,
  "weather": "雨"
}
```

### Scenario Adjuster Logic

**File**: `src/scenario/adjuster.py`

```python
class ScenarioAdjuster:
    def adjust(self, base_predictions, scenario_spec, race_data):
        # 1. Apply pace adjustments by running style
        # 2. Apply bias adjustments by post position
        # 3. Apply track condition adjustments
        # 4. Combine in log-odds space
        # 5. Normalize with Softmax
        return adjusted_predictions
```

---

## Backtesting

### Backtest Engine

**File**: `src/backtest.py`

**Features**:
- Odds-based strategy simulation
- Takeout rate consideration (JRA: ~20-25%)
- Multiple betting strategies
- Kelly criterion support
- Performance metrics

### Running Backtests

```python
from src.backtest import Backtester

backtester = Backtester(
    db_path='netkeiba.db',
    model_dir='models/',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Strategy 1: Bet on EV > 1.0
results = backtester.run_ev_strategy(min_ev=1.0)

# Strategy 2: Top predicted horse
results = backtester.run_top_pick_strategy()

# Strategy 3: Kelly criterion
results = backtester.run_kelly_strategy(kelly_fraction=0.25)

# Print results
print(f"Total bets: {results['n_bets']}")
print(f"Win rate: {results['win_rate']:.2%}")
print(f"ROI: {results['roi']:.2%}")
print(f"Total profit: {results['total_profit']:.0f} yen")
```

### Evaluation Metrics

- **ROI** (Return on Investment): `(total_return - total_bet) / total_bet`
- **Hit Rate**: Fraction of winning bets
- **Average Odds**: Mean odds of bet horses
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline

---

## Explainability (SHAP)

### SHAP Explainer

**File**: `src/shap_explainer.py`

**Purpose**: Generate feature importance explanations for individual predictions

```python
from src.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model_path='models/lgbm_target_win.txt')
explainer.fit(X_train)

# Explain single prediction
shap_values = explainer.explain_one(X_test[0])

# Get text explanation
explanation = explainer.generate_text_explanation(
    shap_values,
    feature_names,
    feature_values
)
print(explanation)
```

**Example Output**:
```
Top factors increasing win probability:
1. win_rate_dist_cat (0.15): Strong historical performance at this distance
2. recent_avg_finish_3 (0.12): Excellent recent form (avg finish: 1.7)
3. hlap_late_vs_race (0.08): Superior closing speed

Top factors decreasing win probability:
1. days_since_last_run (-0.10): Long layoff (120 days)
2. body_weight_diff (-0.05): Weight loss of 8kg
```

### Integration with UI

SHAP explanations are displayed in:
- Horse detail modals (keiba-ui)
- API responses (`/explain/{race_id}/{horse_id}`)
- PDF race reports (future feature)

---

## API

### FastAPI Backend

**File**: `src/api.py`

**Status**: Stub implementation ready for expansion

### Endpoints

#### GET /
Health check
```json
{"status": "ok", "version": "1.0.0"}
```

#### POST /predict
Predict race outcome
```python
# Request
{
  "race_id": "202406050811",
  "include_shap": true,
  "scenario": {
    "pace_type": "M",
    "bias_type": "Flat",
    "track_condition": "良"
  }
}

# Response
{
  "race_id": "202406050811",
  "predictions": [
    {
      "horse_id": "2021101234",
      "horse_name": "ドウデュース",
      "base_pred": 0.25,
      "calibrated_pred": 0.22,
      "delta_baba": 0.01,
      "delta_pace": 0.02,
      "final_prob": 0.28,
      "ev": 1.12,
      "odds": 4.0,
      "rank": 1
    },
    ...
  ],
  "race_sum_prob": 1.0,
  "top_3_horses": ["2021101234", "2020103456", "2019105678"]
}
```

#### GET /explain/{race_id}/{horse_id}
SHAP explanation
```json
{
  "race_id": "202406050811",
  "horse_id": "2021101234",
  "top_features": [
    {"name": "win_rate_dist_cat", "value": 0.35, "shap": 0.15},
    {"name": "recent_avg_finish_3", "value": 1.7, "shap": 0.12}
  ],
  "text_explanation": "Strong distance category performance..."
}
```

#### GET /backtest
Backtest results
```json
{
  "strategy": "ev_threshold",
  "params": {"min_ev": 1.0},
  "period": "2024-01-01 to 2024-12-31",
  "n_bets": 450,
  "win_rate": 0.32,
  "roi": 0.08,
  "total_profit": 145000
}
```

#### GET /races/today
Today's races
```json
{
  "date": "2024-12-16",
  "races": [
    {
      "race_id": "202410050811",
      "place": "中山",
      "name": "ターコイズステークス",
      "start_time": "15:45"
    }
  ]
}
```

### Running API Server

```bash
# Development
python src/api.py

# Production with uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# With workers
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## UI Projects

### keiba-ui (Next.js 14)

**Primary production UI**

**Tech Stack**:
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS

**Structure**:
```
keiba-ui/
├── app/
│   ├── layout.tsx              # Root layout with fonts
│   ├── page.tsx                # Redirects to /scenario
│   ├── globals.css             # Tailwind + custom styles
│   ├── scenario/
│   │   └── page.tsx            # Main scenario analysis page
│   └── components/
│       ├── ScenarioControl.tsx # Weather, pace, bias controls
│       ├── HorseTable.tsx      # Sortable horse list
│       └── HorseDetailModal.tsx # Modal with SHAP & comments
└── public/
    ├── scenario_default.json
    ├── scenario_slow_inner.json
    ├── scenario_middle_flat.json
    └── scenario_high_outer.json
```

**Features**:
- Real-time scenario adjustment
- Sortable columns (probability, EV, odds)
- Horse detail modals with SHAP explanations
- Moisture & cushion value sliders
- Responsive design

**Development**:
```bash
cd keiba-ui
npm install
npm run dev     # http://localhost:3000
npm run build   # Production build
npm run start   # Production server
```

**Environment Variables** (`.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### scenario-ui (React + Vite)

**Alternative lightweight UI**

**Tech Stack**:
- React 18
- Vite
- TypeScript
- CSS Modules

**Development**:
```bash
cd scenario-ui
npm install
npm run dev       # http://localhost:5173
npm run build
npm run preview
```

---

## Development Workflows

### Complete Pipeline

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure netkeiba cookies
cp src/ingestion/.env.example src/.env
# Edit .env with browser cookies

# 3. Scrape race data
cd src
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024

# 4. Build feature table
cd ..
python run_build_features.py

# 5. Train models
python -m src.models.train_lgbm --db netkeiba.db --out models/

# 6. Run backtests
python -c "
from src.backtest import Backtester
bt = Backtester('netkeiba.db', 'models/')
results = bt.run_ev_strategy(min_ev=1.0)
print(results)
"

# 7. Start API server (terminal 1)
python src/api.py

# 8. Start UI (terminal 2)
cd keiba-ui
npm run dev

# 9. Run scenario analysis
python scripts/run_scenario_prediction.py --race-id 202406050811
```

### Incremental Updates

```bash
# Update specific race
python -m ingestion.ingest_runner --race-ids 202410050811 -v

# Rebuild features for updated data
python run_build_features.py

# Retrain models
python -m src.models.train_lgbm --db netkeiba.db --out models/
```

### Debugging Workflows

```bash
# Check database integrity
python diagnose_db.py

# Verify feature table
python check_feature_table.py

# Check horse lap coverage
python scripts/horse_lap_coverage.py

# Drop and rebuild feature table
python drop_feature_table.py
python run_build_features.py
```

---

## Key Conventions

### Code Style

1. **Logging**: Always use `logging` module
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info("Message")
   ```

2. **Type Hints**: Use typing for function signatures
   ```python
   from typing import List, Optional, Dict

   def predict_race(race_id: str, model_path: str) -> Dict[str, float]:
       ...
   ```

3. **Dataclasses**: Use for structured data
   ```python
   from dataclasses import dataclass

   @dataclass
   class Race:
       race_id: str
       date: str
       place: str
   ```

4. **Error Handling**: Explicit try-except with logging
   ```python
   try:
       result = risky_operation()
   except Exception as e:
       logger.error(f"Operation failed: {e}")
       raise
   ```

### Database Operations

1. **Idempotent Upserts**: Use `INSERT OR REPLACE`
   ```python
   conn.execute("""
       INSERT OR REPLACE INTO races (race_id, date, ...)
       VALUES (?, ?, ...)
   """, values)
   ```

2. **Parameterized Queries**: Always use placeholders
   ```python
   # Good
   cursor.execute("SELECT * FROM races WHERE race_id = ?", (race_id,))

   # Bad - SQL injection risk
   cursor.execute(f"SELECT * FROM races WHERE race_id = '{race_id}'")
   ```

3. **Context Managers**: Use `with` for connections
   ```python
   with sqlite3.connect('netkeiba.db') as conn:
       cursor = conn.cursor()
       cursor.execute(...)
   ```

### Feature Engineering

1. **No Future Leakage**: Always filter by datetime
   ```python
   past_runs = all_runs[
       (all_runs['horse_id'] == horse_id) &
       (all_runs['date'] < current_race_date)
   ]
   ```

2. **Null Handling**: Graceful degradation
   ```python
   win_rate = n_wins / n_starts if n_starts > 0 else None
   ```

3. **Vectorization**: Use pandas operations over loops
   ```python
   # Good
   df['win_rate'] = df['n_wins'] / df['n_starts']

   # Bad
   for idx, row in df.iterrows():
       df.loc[idx, 'win_rate'] = row['n_wins'] / row['n_starts']
   ```

### Model Training

1. **Reproducibility**: Set random seeds
   ```python
   import random
   import numpy as np

   random.seed(42)
   np.random.seed(42)
   ```

2. **Train/Val/Test Split**: Use temporal splits
   ```python
   train_end = '2023-12-31'
   val_end = '2024-06-30'
   # Train: before 2024-01-01
   # Val: 2024-01-01 to 2024-06-30
   # Test: after 2024-06-30
   ```

3. **Feature Column Management**: Save and load feature names
   ```python
   import json

   # Save
   with open('feature_columns.json', 'w') as f:
       json.dump(feature_names, f)

   # Load
   with open('feature_columns.json', 'r') as f:
       feature_names = json.load(f)
   ```

### Git Workflow

1. **Never Commit**:
   - `src/netkeiba.db`, `data/*.db`
   - `.env` files
   - `keiba-ui/node_modules/`, `scenario-ui/node_modules/`
   - `__pycache__/`, `*.pyc`
   - Large model files (prefer model registry)

2. **Branch Naming**:
   - `feature/add-xyz`
   - `bugfix/fix-xyz`
   - `refactor/improve-xyz`
   - For automated Claude Code: `claude/claude-md-{session_id}`

3. **Commit Messages**:
   ```
   Add feature: Individual lap time processing

   - Implement horse_laps table parsing
   - Add lap ratio feature computation
   - Update feature_builder.py with hlap_* features
   - Add tests for lap processing
   ```

---

## Important Gotchas

### 1. horse_past_runs Table Dependency

**Issue**: `feature_builder.py` originally assumed `horse_past_runs` table exists, but it's not always populated.

**Solution**: Feature builder now handles missing table gracefully:
```python
if 'horse_past_runs' not in tables or tables['horse_past_runs'].empty:
    # Skip past-run features, use only current race features
    logger.warning("horse_past_runs table empty, skipping historical features")
```

**Action**: When using feature_builder, ensure either:
1. `horse_past_runs` table is populated, OR
2. Accept that historical features will be NULL/skipped

### 2. Lap Ratio Features Require Premium Data

**Issue**: `hlap_*` features depend on `horse_laps` table, which requires netkeiba Premium membership.

**Verification**:
```sql
SELECT COUNT(*) FROM horse_laps;
```

If zero, lap features will be NULL. Model can still train but loses valuable features.

### 3. Race ID Parsing

**Issue**: Race IDs are strings, not integers. Leading zeros matter.

```python
# Correct
race_id = "202406050811"

# Wrong - loses leading zeros
race_id = int("202406050811")  # Becomes 202406050811 (ok in this case)
race_id = int("202401010101")  # Becomes 202401010101 (ok)
race_id = "202401010101"[:-2]  # String manipulation safer
```

### 4. Distance Category Boundaries

**Edge case**: 1800m is boundary between mile and middle
```python
def get_distance_cat(distance: int) -> str:
    if distance < 1400:
        return 'short'
    elif distance < 1800:  # 1800 is middle
        return 'mile'
    elif distance < 2200:
        return 'middle'
    else:
        return 'long'
```

### 5. Track Condition Encoding

**Consistency required** between ingestion and features:
- 良 → 0 or "良" (check your encoding)
- 稍重 → 1 or "稍重"
- 重 → 2 or "重"
- 不良 → 3 or "不良"

**Action**: Always use same encoding throughout pipeline.

### 6. Softmax Normalization Requirement

**Issue**: After applying log-odds adjustments, probabilities may not sum to 1.

**Solution**: Always apply Softmax normalization:
```python
from scipy.special import softmax

log_odds = [logit(p) + adjustment for p in base_probs]
normalized_probs = softmax(log_odds)
assert abs(sum(normalized_probs) - 1.0) < 1e-6
```

### 7. Odds vs Probability Confusion

**Odds**: Betting odds (e.g., 4.0 = 4倍 = 400円 return on 100円 bet)
**Probability**: Win probability (e.g., 0.25 = 25%)

**Conversion**:
```python
# Odds to probability (ignoring takeout)
prob = 1 / odds

# Probability to fair odds
odds = 1 / prob

# Probability to market odds (with ~20% takeout)
market_odds = 1 / (prob * 0.8)
```

### 8. Date Format Consistency

**Database**: `YYYY-MM-DD` (ISO 8601)
**Display**: `YYYY年MM月DD日` (Japanese format)

```python
from datetime import datetime

# Parse
date_obj = datetime.strptime('2024-12-16', '%Y-%m-%d')

# Format for DB
db_date = date_obj.strftime('%Y-%m-%d')

# Format for display
display_date = date_obj.strftime('%Y年%m月%d日')
```

### 9. Memory Issues with Large Feature Tables

**Issue**: Loading entire feature table into memory can OOM on large datasets.

**Solution**: Use chunked reading or SQL filtering:
```python
# Chunk reading
for chunk in pd.read_sql_query(query, conn, chunksize=10000):
    process(chunk)

# Or filter by date range
query = """
    SELECT * FROM feature_table
    WHERE race_id LIKE '2024%'
"""
```

### 10. Timezone Handling

**Issue**: Race times are JST (Japan Standard Time, UTC+9)

```python
import pytz

jst = pytz.timezone('Asia/Tokyo')
race_time_jst = datetime(2024, 12, 16, 15, 45, tzinfo=jst)
race_time_utc = race_time_jst.astimezone(pytz.utc)
```

---

## Testing

### Current State

**Status**: Minimal formal testing infrastructure
- No `pytest` configuration
- No `tests/` directory
- Ad-hoc test scripts

**Test Files**:
- `test_horse_past.py` - Horse past data parsing
- `src/test_calibration.py` - Calibration methods

### Running Tests

```bash
# Individual test scripts
python test_horse_past.py
python src/test_calibration.py

# Diagnostic scripts
python diagnose_db.py
python check_feature_table.py
```

### Future Testing Strategy

**Recommended**:
1. Add `pytest` to requirements.txt
2. Create `tests/` directory structure:
   ```
   tests/
   ├── test_ingestion.py
   ├── test_features.py
   ├── test_models.py
   ├── test_scenario.py
   └── fixtures/
       └── sample_data.db
   ```
3. Add GitHub Actions CI/CD

**Example Test**:
```python
# tests/test_features.py
import pytest
from src.features.feature_builder import build_features_for_race

def test_feature_builder_no_leakage():
    """Ensure no future information in features"""
    race_date = '2024-06-01'
    features = build_features_for_race(race_id, tables, race_date)

    # All historical stats should be from before race_date
    assert all(features['past_run_dates'] < race_date)
```

---

## Performance Optimization

### Database Indexes

**Recommended indexes** for `netkeiba.db`:

```sql
-- Race lookups
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_place ON races(place);

-- Horse lookups
CREATE INDEX IF NOT EXISTS idx_race_results_horse ON race_results(horse_id);
CREATE INDEX IF NOT EXISTS idx_race_results_date ON race_results(race_id);

-- Feature table
CREATE INDEX IF NOT EXISTS idx_feature_table_race ON feature_table(race_id);
CREATE INDEX IF NOT EXISTS idx_feature_table_horse ON feature_table(horse_id);

-- Lap times
CREATE INDEX IF NOT EXISTS idx_horse_laps_horse ON horse_laps(horse_id);
CREATE INDEX IF NOT EXISTS idx_horse_laps_race ON horse_laps(race_id);
```

### Feature Computation Caching

**Strategy**: Cache race attributes to avoid repeated lookups

```python
# In feature_builder.py
race_attrs_cache = {}

def get_race_attrs(race_id, tables):
    if race_id in race_attrs_cache:
        return race_attrs_cache[race_id]

    attrs = compute_race_attrs(race_id, tables)
    race_attrs_cache[race_id] = attrs
    return attrs
```

### Model Prediction Batching

**Issue**: Predicting one horse at a time is slow

**Solution**: Batch predictions by race
```python
# Good - batch prediction
X_race = feature_table[feature_table['race_id'] == race_id]
predictions = model.predict_proba(X_race)

# Bad - row-by-row
for _, row in X_race.iterrows():
    pred = model.predict_proba([row.values])  # Slow!
```

---

## Deployment Considerations

### Environment Variables

**Production `.env` template**:
```bash
# Database
DATABASE_PATH=/data/netkeiba.db

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Models
MODEL_DIR=/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=/logs/keiba.log

# netkeiba credentials
NETKEIBA_COOKIE_netkeiba=...
NETKEIBA_COOKIE_nkauth=...
```

### Docker (Future)

**Recommended Dockerfile**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/

# Run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring

**Key Metrics**:
- Model prediction latency
- Database query times
- API response times
- Feature generation time
- Backtest ROI
- Model AUC/Log Loss

**Tools** (future integration):
- MLflow for experiment tracking
- Prometheus + Grafana for monitoring
- Sentry for error tracking

---

## Resources

### Internal Documentation

- `README.md` - Japanese project overview
- `AGENT.md` - Development agent guidelines
- `GIT_SETUP.md` - Git workflow instructions
- `src/ingestion/README.md` - Ingestion pipeline details

### External Resources

**JRA Official**:
- https://www.jra.go.jp/ - Japan Racing Association

**netkeiba**:
- https://www.netkeiba.com/ - Public site
- https://db.netkeiba.com/ - Database (Premium required)

**Machine Learning**:
- LightGBM: https://lightgbm.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- scikit-learn: https://scikit-learn.org/

**Web Development**:
- Next.js: https://nextjs.org/docs
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/

---

## FAQ

### Q: Why is feature_table empty after running build_feature_table()?

**A**: Common causes:
1. `horse_past_runs` table doesn't exist - builder may skip all features
2. No races in database - check `SELECT COUNT(*) FROM races`
3. No race_results - check `SELECT COUNT(*) FROM race_results`
4. Date filtering too strict - check race dates vs current date

**Debug**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
from features import build_feature_table
# Check debug logs for specific errors
```

### Q: Why are hlap_* features all NULL?

**A**: Requires `horse_laps` table populated. Check:
```sql
SELECT COUNT(*) FROM horse_laps;
```

If zero, you need to:
1. Have netkeiba Premium membership
2. Re-run ingestion with lap time fetching enabled

### Q: How do I add a new feature?

**A**:
1. Edit `src/features/feature_builder.py`
2. Add feature computation in `build_features_for_race()`
3. Add column to `FEATURE_TABLE_DDL` in `sqlite_store_feature.py`
4. Drop and rebuild feature_table
5. Update `feature_columns.json` if training new model

### Q: Can I use this for NAR (local/regional) races?

**A**: Current implementation is JRA-specific. NAR support requires:
1. Update `race_list.py` for NAR race IDs
2. Update parsers for NAR HTML structure
3. Adjust feature engineering for NAR characteristics
4. Separate models (JRA and NAR are very different)

### Q: How do I handle missing data in production?

**A**:
1. **Race data missing**: Use fallback to average/median features
2. **Horse past data missing**: Set historical stats to NULL, use only current race features
3. **Lap data missing**: hlap_* features → NULL, model uses other features
4. **Always**: Log missing data events for monitoring

### Q: What's the expected prediction accuracy?

**A**: Realistic expectations:
- **Win prediction AUC**: 0.75-0.85 (good)
- **Top 3 prediction AUC**: 0.70-0.80 (good)
- **Hit rate** (picking winner): 15-30% (JRA average ~10%)
- **ROI**: Positive ROI with good calibration and scenario tuning

Horse racing is inherently unpredictable - aim for long-term edge, not perfect predictions.

### Q: How often should I retrain models?

**A**:
- **Development**: After significant feature changes
- **Production**: Monthly or quarterly
- **Triggers**: Performance degradation, major rule changes, new data available

**Process**:
```bash
# 1. Ingest latest data
python -m ingestion.ingest_runner --start-year 2024 --end-year 2024

# 2. Rebuild features
python run_build_features.py

# 3. Retrain
python -m src.models.train_lgbm --db netkeiba.db --out models/

# 4. Backtest new model
python -c "from src.backtest import Backtester; ..."

# 5. Compare to old model, deploy if better
```

---

## Contributors

- **obn** - Project creator
- **Claude (Anthropic)** - AI development assistant
- **ChatGPT (OpenAI)** - AI development assistant

---

## License

Not specified - private project

---

## Version History

- **v3.3.1** (2024-12-14) - Lap ratio features, feature_builder refactor
- **v3.0** (2024-12-04) - Calibration v3, scenario system
- **v2.0** (2024-11) - LightGBM models, backtest engine
- **v1.0** (2024-10) - Initial ingestion pipeline

---

**Last Updated**: 2025-12-16
**Document Version**: 1.0
