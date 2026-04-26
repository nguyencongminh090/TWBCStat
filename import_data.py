"""
TWBC 2026 — CSV → SQLite Importer
===================================
Reads matches.csv and game_results.csv,
normalizes the data, and populates the BCNF schema.

Usage:
    python src/import_data.py --csv data/raw --db data/processed/twbc.db
"""

import csv
import sqlite3
import argparse
import os
import re

# Known team names — same list as in crawler.py
KNOWN_TEAMS = sorted([
    "Poland A", "Poland B", "Poland C",
    "Hungary A", "Hungary B", "Hungary C",
    "Czechia A", "Czechia B", "Czechia C", "Czechia D", "Czechia E", "Czechia F",
    "Czech Republic A", "Czech Republic B", "Czech Republic C",
    "Czech Republic D", "Czech Republic E", "Czech Republic F",
    "Slovakia", "Russia", "China",
    "Team International A", "Team International B",
    "Team International C", "Team International D",
], key=len, reverse=True)


def clean_player_name(raw: str) -> str:
    """Strip team-name prefixes and letter-code prefixes that bleed into names.

    Examples:
        "International A  Ashot Avetisyan" → "Ashot Avetisyan"
        "C  Stanislav Hyžík jr."          → "Stanislav Hyžík jr."
        "International D  Nguyễn Công Minh" → "Nguyễn Công Minh"
    """
    name = raw.strip()

    # Strip full team name prefixes (longest first)
    for team in KNOWN_TEAMS:
        if name.startswith(team):
            name = name[len(team):].strip()
            break

    # Strip residual letter-code prefixes like "A  ", "B  ", "C  ", "D  "
    name = re.sub(r'^[A-Z]\s{2,}', '', name)

    # Strip partial suffixes like "International A  " or "International D  "
    name = re.sub(r'^International\s+[A-D]\s+', '', name)

    return name.strip()


def read_csv(path):
    """Read a CSV file and return a list of dicts."""
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def import_data(csv_dir, db_path):
    matches_csv = os.path.join(csv_dir, 'matches.csv')
    games_csv = os.path.join(csv_dir, 'game_results.csv')

    matches_data = read_csv(matches_csv)
    games_data = read_csv(games_csv)

    # Remove DB if it already exists (fresh import)
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create schema
    # Schema lives in ../sql/ relative to this script (src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(project_root, 'sql', 'schema.sql')
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    with open(schema_path, 'r', encoding='utf-8') as f:
        conn.executescript(f.read())

    cur = conn.cursor()

    # ── Step 1: Collect all unique teams ──────────────────
    team_names = set()
    for row in matches_data:
        team_names.add(row['team_a'])
        team_names.add(row['team_b'])
    for row in games_data:
        team_names.add(row['player_a_team'])
        team_names.add(row['player_b_team'])
    team_names.discard('')

    team_id_map = {}
    for name in sorted(team_names):
        cur.execute("INSERT INTO teams (name) VALUES (?)", (name,))
        team_id_map[name] = cur.lastrowid

    print(f"  teams: {len(team_id_map)} inserted")

    # ── Step 2: Collect all unique players ────────────────
    # Build nick → (best_name, team) from game_results
    player_info = {}  # nick → {'name': str, 'team': str}

    for row in games_data:
        nick_a = row['player_a_nick']
        nick_b = row['player_b_nick']
        name_a = clean_player_name(row['player_a_name'])
        name_b = clean_player_name(row['player_b_name'])
        team_a = row['player_a_team']
        team_b = row['player_b_team']

        if nick_a and nick_a not in player_info:
            player_info[nick_a] = {'name': name_a, 'team': team_a}
        elif nick_a and name_a and len(name_a) > len(player_info.get(nick_a, {}).get('name', '')):
            # Keep the longer (more complete) name variant
            player_info[nick_a]['name'] = name_a

        if nick_b and nick_b not in player_info:
            player_info[nick_b] = {'name': name_b, 'team': team_b}
        elif nick_b and name_b and len(name_b) > len(player_info.get(nick_b, {}).get('name', '')):
            player_info[nick_b]['name'] = name_b

    player_id_map = {}
    for nick, info in sorted(player_info.items()):
        team = info['team']
        tid = team_id_map.get(team)
        if tid is None:
            print(f"    ⚠ Player '{nick}' has unknown team '{team}', skipping")
            continue
        cur.execute(
            "INSERT INTO players (nick, full_name, team_id) VALUES (?, ?, ?)",
            (nick, info['name'], tid)
        )
        player_id_map[nick] = cur.lastrowid

    print(f"  players: {len(player_id_map)} inserted")

    # ── Step 3: Insert matches ────────────────────────────
    match_id_map = {}  # (tournament_round, team_a, team_b) → match_id

    for row in matches_data:
        t_round = int(row['tournament_round'])
        ta_id = team_id_map[row['team_a']]
        tb_id = team_id_map[row['team_b']]
        result = row['match_result']
        url = row.get('url', row.get('source_url', ''))

        cur.execute(
            "INSERT INTO matches (tournament_round, team_a_id, team_b_id, match_result, source_url) "
            "VALUES (?, ?, ?, ?, ?)",
            (t_round, ta_id, tb_id, result, url)
        )
        mid = cur.lastrowid
        match_key = f"{t_round}:{row['team_a']} vs {row['team_b']}"
        match_id_map[match_key] = mid

    print(f"  matches: {len(match_id_map)} inserted")

    # ── Step 4: Insert sub_rounds and pairings ────────────
    sub_round_id_map = {}  # (match_id, round_number) → sub_round_id
    pairing_count = 0

    for row in games_data:
        t_round = int(row['tournament_round'])
        match_id_str = row['match_id']  # e.g. "Poland A vs Czechia E"
        match_key = f"{t_round}:{match_id_str}"

        mid = match_id_map.get(match_key)
        if mid is None:
            print(f"    ⚠ No match found for key '{match_key}', skipping row")
            continue

        sr_num = int(row['sub_round'])
        sr_key = (mid, sr_num)

        if sr_key not in sub_round_id_map:
            cur.execute(
                "INSERT INTO sub_rounds (match_id, round_number) VALUES (?, ?)",
                (mid, sr_num)
            )
            sub_round_id_map[sr_key] = cur.lastrowid

        sr_id = sub_round_id_map[sr_key]

        pa_id = player_id_map.get(row['player_a_nick'])
        pb_id = player_id_map.get(row['player_b_nick'])
        if pa_id is None or pb_id is None:
            print(f"    ⚠ Unknown player nick in pairing: "
                  f"{row['player_a_nick']}={pa_id}, {row['player_b_nick']}={pb_id}")
            continue

        cur.execute(
            "INSERT INTO pairings (sub_round_id, player_a_id, player_b_id, score_a, score_b) "
            "VALUES (?, ?, ?, ?, ?)",
            (sr_id, pa_id, pb_id, float(row['score_a']), float(row['score_b']))
        )
        pairing_count += 1

    print(f"  sub_rounds: {len(sub_round_id_map)} inserted")
    print(f"  pairings: {pairing_count} inserted")

    conn.commit()

    # ── Step 5: Verify with views ─────────────────────────
    print("\n── Verification ──")

    cur.execute("SELECT COUNT(*) FROM v_matches")
    print(f"  v_matches: {cur.fetchone()[0]} rows")

    cur.execute("SELECT COUNT(*) FROM v_player_match_summary")
    print(f"  v_player_match_summary: {cur.fetchone()[0]} rows")

    cur.execute("SELECT COUNT(*) FROM v_player_overall")
    print(f"  v_player_overall: {cur.fetchone()[0]} rows")

    print("\n── Top 10 players (min 24 games) ──")
    cur.execute("""
        SELECT player_nick, player_name, team, total_score, total_games, efficiency
        FROM v_player_overall
        WHERE total_games >= 24
        ORDER BY efficiency DESC
        LIMIT 10
    """)
    print(f"  {'Nick':<18} {'Name':<25} {'Team':<22} {'Score':>6} {'Games':>6} {'Eff':>7}")
    print(f"  {'─'*18} {'─'*25} {'─'*22} {'─'*6} {'─'*6} {'─'*7}")
    for row in cur.fetchall():
        nick, name, team, score, games, eff = row
        print(f"  {nick:<18} {name:<25} {team:<22} {score:>6.1f} {games:>6.0f} {eff:>7.4f}")

    conn.close()
    print(f"\n✓ Database saved to {db_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Import TWBC CSV data into normalized SQLite DB")
    ap.add_argument("--csv", default="data/raw", help="Directory with CSV files")
    ap.add_argument("--db", default="data/processed/twbc.db", help="Output SQLite database path")
    args = ap.parse_args()
    import_data(args.csv, args.db)
