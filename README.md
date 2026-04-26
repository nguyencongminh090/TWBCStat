# TWBCStat — Team World Blitz Championship 2026 Statistics

Data pipeline and analysis toolkit for the [TWBC 2026](https://sites.google.com/view/worldblitzcup/twbc-2026) tournament.

## Project Structure

```
TWBCStat/
├── src/                    # Source code
│   ├── crawler.py          # Web scraper: fetches match pages → CSV
│   └── import_data.py      # Normalizer: CSV → SQLite (BCNF)
├── sql/
│   └── schema.sql          # Database schema (5 tables + 5 views)
├── data/
│   ├── raw/                # Crawler output (CSV files)
│   │   ├── matches.csv
│   │   ├── game_results.csv
│   │   └── player_match_summary.csv
│   └── processed/          # Normalized SQLite database
│       └── twbc.db
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Test scripts
└── README.md
```

## Pipeline

```
 Website  ──crawler.py──▶  data/raw/*.csv  ──import_data.py──▶  data/processed/twbc.db
```

### 1. Crawl match data
```bash
python src/crawler.py --out data/raw
```

### 2. Import into normalized database
```bash
python src/import_data.py --csv data/raw --db data/processed/twbc.db
```

### 3. Query the database
```bash
sqlite3 data/processed/twbc.db "SELECT * FROM v_player_overall ORDER BY efficiency DESC LIMIT 10;"
```

## Database Schema

Five normalized tables (BCNF), five computed views:

| Layer | Tables | Purpose |
|---|---|---|
| **Stored** | `teams`, `players`, `matches`, `sub_rounds`, `pairings` | Source facts only |
| **Views** | `v_pairings`, `v_sub_rounds`, `v_matches`, `v_player_match_summary`, `v_player_overall` | All derived metrics |

## Requirements

```
pip install requests pandas
```
