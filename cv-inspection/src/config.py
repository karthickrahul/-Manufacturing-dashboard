"""Every ASSUMPTION used by the simulations and the impact model lives here, in one place.

Nothing in this file is measured from NEU-DET unless stated. Replace with plant data before
quoting any money figure as anything but "modelled".
"""

# ---- production line (assumed) ----
PARTS_PER_DAY = 5000
HOURS_PER_DAY = 16                       # two 8-hour shifts
WORKING_DAYS = 250
BATCH_SIZE = 200                         # parts per SPC batch
PARTS_PER_HOUR = PARTS_PER_DAY / HOURS_PER_DAY
BATCH_HOURS = BATCH_SIZE / PARTS_PER_HOUR

# ---- process (assumed) ----
P0 = 0.03                                # in-control defect rate
P1 = 0.07                                # defect rate after a process shift (e.g. tool wear)
N_BATCHES = 60
BASELINE_BATCHES = 20                    # Phase I period that sets control limits
SHIFT_AT = 31                            # first out-of-control batch (1-indexed)

# ---- inspection strategies (assumed unless noted) ----
MANUAL_SAMPLE = 20                       # parts a human inspects per batch under sampling inspection
MISS_MANUAL = 0.20                       # share of defects a human inspector misses (mode)
FALSE_ALARM_MANUAL = 0.01
FALSE_ALARM_AI = 0.02                    # share of GOOD parts the AI wrongly flags. NOT measurable on
                                         # NEU-DET (no good images) -> pure assumption
# AI part-level recall is MEASURED (results/bootstrap_flag_recall.csv); not set here.

# ---- economics (assumed, EUR) ----
LABOUR_EUR_H = 25
T_MANUAL_S = (8, 12, 20)                 # triangular (low, mode, high): seconds to inspect a part by eye
T_REVIEW_S = (10, 15, 25)                # seconds for a human to re-check a flagged part
COST_ESCAPE = (50, 100, 300)             # cost of one defective part reaching the customer
SCRAP_PART = (5, 15, 40)                 # scrap cost per defective part produced
DEFECT_RATE = (0.02, 0.03, 0.08)         # daily defect-rate uncertainty (mode = P0)
SHIFTS_PER_YEAR = (2, 6, 12)
SYSTEM_EUR_DAY = 40                      # camera + compute + upkeep, amortised
# MISS_MANUAL uncertainty in the Monte Carlo
MISS_MANUAL_RANGE = (0.10, 0.20, 0.30)

# ---- defect mix of the simulated demo stream (assumed) ----
MIX = {"crazing": 0.10, "inclusion": 0.25, "patches": 0.10,
       "pitted_surface": 0.10, "rolled-in_scale": 0.15, "scratches": 0.30}
