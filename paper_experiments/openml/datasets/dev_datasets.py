"""Development datasets from the TabArena paper.

Source: Supplementary Tables 5 & 6 of
  "TabArena: A Living Benchmark for Machine Learning on Tabular Data"
  (Nature, 2024) — https://doi.org/10.1038/s41586-024-08328-6

These are OpenML *dataset* IDs, not task IDs.
Use `get_dev_task_ids()` to resolve them to OpenML task IDs at runtime.

Notes
-----
- Only the first 10 000 rows of each dataset are used.
- Datasets with >500 features or >10 classes are excluded.
- Overlap with the TabArena test set has been removed.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Table 5: Classification development datasets
# ---------------------------------------------------------------------------

# Override: preferred task IDs for datasets where the oldest task isn't 10-fold CV.
# Maps dataset_id (did) -> preferred task_id (tid).
PREFERRED_TASK_IDS: dict[int, int] = {
    338: 3497,   # grub-damage:    oldest task is 1-fold; this one is 10-fold
    340: 3499,   # squash-stored:  oldest task is 1-fold; this one is 10-fold
}

CLASSIFICATION_DATASET_IDS: dict[str, int] = {
    "adaprior":                             1037,
    "allbp":                               40707,
    "analcatdata_asbestos":                  459,
    "analcatdata_cyyoung8092":               465,
    "ar3":                                  1060,
    "ar6":                                  1064,
    "baseball":                              185,
    "birds":                               40588,
    "blogger":                              1463,
    "breast-tissue":                        1465,
    "cjs":                                   473,
    "clean1":                              40665,
    "cleveland-nominal":                   40711,
    "compas-two-years":                    42192,
    "confidence":                            468,
    "corral":                              40669,
    "delta_ailerons":                        803,
    "dermatology":                            35,
    "eye_movements":                       44130,
    "GAMETES_Epistasis":                   40646,
    "grub-damage":                           338,
    "heart-c":                                49,
    "heart-h":                                51,
    "heart-statlog":                          53,
    "heloc":                               45026,
    "hill-valley":                          1479,
    "ionosphere":                             59,
    "JapaneseVowels":                        375,
    "jungle_chess_2pcs_endgame_panther_lion": 40997,
    "led24":                               40677,
    "mc1":                                  1056,
    "Midwest_Survey_nominal":              42532,
    "mw1":                                  1071,
    "national-longitudinal-survey-binary": 43892,
    "olindda_outliers":                    42793,
    "page-blocks":                            30,
    "penguins":                            42585,
    "regime_alimentaire":                  42172,
    "ringnorm":                             1496,
    "rl":                                  44160,
    "rmftsa_sleepdata":                      679,
    "scene":                                 312,
    "Smartphone-Based_Recognition":         4153,
    "squash-stored":                         340,
    "synthetic_control":                     377,
    "teachingAssistant":                    1115,
    "thyroid-allbp":                       40474,
    "thyroid-ann":                         40497,
    "Titanic":                             40704,
    "tokyo1":                              40705,
    "Tour-and-Travels-Churn":             45545,
    "user-knowledge":                       1508,
    "volcanoes-a1":                         1527,
    "vote":                                   56,
    "waveform-5000":                          60,
    "wholesale-customers":                  1511,
    "xd6":                                 40693,
}

# ---------------------------------------------------------------------------
# Table 6: Regression development datasets
# ---------------------------------------------------------------------------

REGRESSION_DATASET_IDS: dict[str, int] = {
    "AAPL_stock_price_2021_2022":        43878,
    "ames_housing":                      43926,
    "analcatdata_apnea3":                  555,
    "analcatdata_birthday":                456,
    "analcatdata_gviolence":               518,
    "analcatdata_michiganacc":             551,
    "analcatdata_supreme":               44055,
    "arsenic-male-bladder":                482,
    "autoMpg":                             196,
    "autoPrice":                           207,
    "balloon":                             512,
    "bodyfat":                             560,
    "boston_corrected":                    543,
    "chatfield_4":                         695,
    "chscase_vine2":                       689,
    "cloud":                               210,
    "cocomo_numeric":                     1051,
    "CPMP-2015-runtime-regression":      41928,
    "cps_85_wages":                        534,
    "cpu_act":                           44132,
    "cpu_small":                           227,
    "crimecommunitynums":                41968,
    "crimecommunitynums2":               41969,
    "debutanizer":                       23516,
    "DEE":                               42360,
    "delta_elevators":                     198,
    "ELE-2":                             42362,
    "fishcatch":                           232,
    "GeographicalOriginalofMusic":        4544,
    "kc1-numeric":                        1070,
    "kdd_coil_1":                          567,
    "kidney":                              553,
    "laser":                             42364,
    "mauna-loa-atmospheric-co2":         41187,
    "meta":                                566,
    "MIP-2016-PAR10-regression":         41938,
    "mtp":                                 405,
    "plasma_retinol":                      511,
    "pm10":                                522,
    "pollen":                              529,
    "puma32H":                             308,
    "QSAR_fish_toxycity":                44028,
    "rabe_265":                            660,
    "rabe_266":                            663,
    "RAM_price":                         40601,
    "residential_building":              42366,
    "s1":                                42110,
    "stock_fardamento02":                42545,
    "treasury":                          42367,
    "triazines":                           206,
    "visualizing_soil":                  44056,
    "weather_ankara":                    42368,
    "weather_izmir":                     42369,
    "wind":                                503,
}
