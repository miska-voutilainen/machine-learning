# Helsinki city bikes: station roles, weather and demand

Project work for *Data handling and machine learning* (Metropolia UAS).
The analysis follows the CRISP-DM process and is reported in
[`citybike_analysis.ipynb`](citybike_analysis.ipynb).

## Research questions

1. **Station roles** – can stations be grouped by when and in which direction bikes move there? (k-means, Ward)
2. **Weather in forecasting** – how much does weather improve an hourly demand forecast for an unseen season? (random forest, log-linear regression)
3. **Rain sensitivity** – do some station roles or trip purposes react more strongly to rain? (regression with calendar controls)

## Data

All sources are open data (CC BY 4.0) and are downloaded by the notebook on the first run:

- HSL city bike origin-destination trips, April–October 2024 and 2025 (~5.1 million trips)
- HSL city bike station register (Helsinki Region Infoshare)
- FMI hourly weather observations, Helsinki Kaisaniemi

## Running

```bash
pip install -r requirements.txt
jupyter notebook citybike_analysis.ipynb
```

The first run downloads about 450 MB into `data/raw/` (not committed). A full run takes a few minutes.
Small processed tables are written to `data/processed/` and the figures used in the seminar slides to `figures/`.

## Contents

| Path | |
|---|---|
| `citybike_analysis.ipynb` | Analysis report (CRISP-DM) |
| `src/data_sources.py` | Download helpers for trips, stations and weather |
| `data/processed/` | Aggregated tables produced by the notebook |
| `figures/` | Figures exported by the notebook |
| `slides/citybike_seminaari.pptx` | Seminar presentation (in Finnish), split between Elias E, Elias N, Veijo and Miska |
