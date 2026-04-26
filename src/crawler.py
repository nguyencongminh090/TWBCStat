"""
TWBC 2026 - Match Data Crawler  v3
=====================================
Fixes vs v2:
  - Content extraction: use LAST occurrence of tournament title
    (nav bar also contains the title → search() was picking up nav, not body)
  - Player name cleanup: strip team name prefixes bleeding into names
  - Summary regex: require uppercase start, exclude colons/digits in name

Chạy từ máy local:
    pip install requests pandas
    python src/crawler.py --out data/raw
"""

import re, time, os, argparse
import requests
import pandas as pd
from dataclasses import dataclass, asdict

BASE = "https://sites.google.com/view/worldblitzcup"

# Match URLs will be discovered dynamically

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Known team names — used to strip prefix pollution in player names
KNOWN_TEAMS = sorted([
    "Poland A", "Poland B", "Poland C",
    "Hungary A", "Hungary B", "Hungary C",
    "Czechia A", "Czechia B", "Czechia C", "Czechia D", "Czechia E", "Czechia F",
    "Czech Republic A", "Czech Republic B", "Czech Republic C",
    "Czech Republic D", "Czech Republic E", "Czech Republic F",
    "Slovakia", "Russia", "China",
    "Team International A", "Team International B",
    "Team International C", "Team International D",
], key=len, reverse=True)   # longest first for prefix matching

# ── Data classes ──────────────────────────────────────────

@dataclass
class GameResult:
    tournament_round: int
    match_id: str
    sub_round: int
    sub_round_score_a: float
    sub_round_score_b: float
    player_a_name: str
    player_a_nick: str
    player_a_team: str
    player_b_name: str
    player_b_nick: str
    player_b_team: str
    score_a: float
    score_b: float
    total_games: float
    winner: str
    wins: int = None
    draws: int = None
    losses: int = None

@dataclass
class PlayerMatchSummary:
    tournament_round: int
    match_id: str
    player_name: str
    player_nick: str
    team: str
    total_score: float
    total_games: float
    efficiency: float

@dataclass
class Match:
    tournament_round: int
    team_a: str
    team_b: str
    score_a: float
    score_b: float
    match_result: str
    match_winner: str
    num_sub_rounds: int
    url: str

# ── Helpers ───────────────────────────────────────────────

def fetch(url, session):
    r = session.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text

def strip_flags(text):
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]{2}', '', text)   # national flags
    text = re.sub(r'[\U0001F30D-\U0001F30F]', '', text)      # globe emoji
    return text

def winner(a, b):
    return "A" if a > b else ("B" if b > a else "Draw")

def clean_name(raw: str) -> str:
    """Strip team-name prefix that bleeds into player names in summary section."""
    name = raw.strip()
    for team in KNOWN_TEAMS:
        if name.startswith(team):
            name = name[len(team):].strip()
            break
    return name

def parse_playok_html(html):
    wins, draws, losses = 0, 0, 0
    for tr_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE):
        tr_content = tr_match.group(1)
        tds = re.findall(r'<td[^>]*>(.*?)</td>', tr_content, re.DOTALL | re.IGNORECASE)
        if len(tds) >= 3:
            res = tds[2].strip().lower()
            res = re.sub(r'<[^>]+>', '', res).strip()
            if res == 'win': wins += 1
            elif res == 'loss': losses += 1
            elif res == 'draw': draws += 1
    return wins, draws, losses

# ── Main parser ───────────────────────────────────────────

def parse_page(raw_html, t_round, url, session):
    playok_links = {}
    for m in re.finditer(r'href="(https://www\.playok\.com/en/stat\.phtml\?u=([^&"]+)&[^"]*oid=([^&"]+)[^"]*)"', raw_html):
        p_url = m.group(1).replace('&amp;', '&')
        player1 = m.group(2).lower()
        player2 = m.group(3).lower()
        playok_links[(player1, player2)] = (p_url, player1)
        playok_links[(player2, player1)] = (p_url, player1)

    # 1. Strip HTML tags → plain text, collapse whitespace
    text = re.sub(r'<[^>]+>', ' ', raw_html)
    text = re.sub(r'\s+', ' ', text)
    text = strip_flags(text)

    # 2. Determine team names from URL slug to avoid nav bar pollution
    # Strip query parameters (e.g. ?authuser=0) before slug extraction
    clean_url = url.split("?")[0]
    slug = clean_url.split("/")[-1]
    slug_fixed = slug.replace("-vsteam-", "-vs-team-").replace("vsteam-", "vs-team-")
    if "-vs-" in slug_fixed:
        slug_a, slug_b = slug_fixed.split("-vs-", 1)
    else:
        slug_a, slug_b = slug_fixed, slug_fixed

    def slug_to_team(s):
        s_clean = s.replace("-", "").lower()
        for t in KNOWN_TEAMS:
            if t.replace(" ", "").lower() == s_clean:
                return t
        return s.replace("-", " ").title()

    team_a = slug_to_team(slug_a)
    team_b = slug_to_team(slug_b)
    match_id = f"{team_a} vs {team_b}"

    # 3. Parse match header score from raw_html (sometimes it's only in the meta og:description)
    score_match = re.search(r'([\d.½/]+)\s*:\s*([\d.½/]+)\s*(?:<[^>]+>\s*)*\(\s*([\d.]+)\s*:\s*([\d.]+)\s*\)', raw_html)
    if not score_match:
        with open("debug_html.txt", "w", encoding="utf-8") as f:
            f.write(raw_html)
        with open("debug_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"    ! Cannot parse match header score in {url}")
        return None, [], []

    match_result = f"{score_match.group(1)}:{score_match.group(2)}"
    score_a = float(score_match.group(3))
    score_b = float(score_match.group(4))

    # We no longer need to slice the nav bar away because SR_RE and SUM_RE 
    # look for specific keywords (ROUND, Final result) that don't appear in the nav bar.
    content = text

    # Trim at page footer
    footer_m = re.search(r'Google Sites|Report abuse', content)
    if footer_m:
        content = content[:footer_m.start()]

    # 4. Parse sub-rounds
    SR_RE = re.compile(
        r'ROUND\s+(\d+)\s+\(([\d.]+):([\d.]+)\)'
        r'(.*?)'
        r'(?=ROUND\s+\d+\s+\(|Final)',
        re.DOTALL
    )
    # Pairing: anchor tags stripped leave spaces around nicks → \(\s*nick\s*\)
    # Name group excludes '(' and ')' to prevent capturing across pairing boundaries
    PAIR_RE = re.compile(
        r'([^()]{2,40}?)'
        r'\(\s*(\w+)\s*\)'
        r'\s+vs\s+'
        r'([^()]{2,40}?)'
        r'\(\s*(\w+)\s*\)'
        r'\s+([\d.]+):([\d.]+)'
    )

    game_results = []
    sub_rounds_found = list(SR_RE.finditer(content))

    for srm in sub_rounds_found:
        sr_num = int(srm.group(1))
        sr_sa  = float(srm.group(2))
        sr_sb  = float(srm.group(3))
        sr_txt = srm.group(4)

        for pm in PAIR_RE.finditer(sr_txt):
            pa_name = pm.group(1).strip()
            pa_nick = pm.group(2).strip()
            pb_name = pm.group(3).strip()
            pb_nick = pm.group(4).strip()
            ga = float(pm.group(5))
            gb = float(pm.group(6))

            game_result = GameResult(
                tournament_round   = t_round,
                match_id           = match_id,
                sub_round          = sr_num,
                sub_round_score_a  = sr_sa,
                sub_round_score_b  = sr_sb,
                player_a_name      = pa_name,
                player_a_nick      = pa_nick,
                player_a_team      = team_a,
                player_b_name      = pb_name,
                player_b_nick      = pb_nick,
                player_b_team      = team_b,
                score_a            = ga,
                score_b            = gb,
                total_games        = ga + gb,
                winner             = winner(ga, gb),
            )
            
            link_info = playok_links.get((pa_nick.lower(), pb_nick.lower()))
            if link_info:
                p_url, u_player = link_info
                try:
                    p_html = fetch(p_url, session)
                    w, d, l = parse_playok_html(p_html)
                    if pa_nick.lower() != u_player:
                        w, l = l, w
                    game_result.wins = w
                    game_result.draws = d
                    game_result.losses = l
                    time.sleep(0.5)
                except Exception as e:
                    print(f"      ! Failed to fetch playok for {pa_nick} vs {pb_nick}")
            
            game_results.append(game_result)

    # 5. Parse player summaries after "Final result"
    # FIX v3: require name starts uppercase, no colons/digits in name
    SUM_RE = re.compile(
        r'(?:^|\s)'
        r'([A-Z\u00C0-\u024F][^(:\d]{1,35}?)'   # name: uppercase start, no : or digits
        r'\(\s*(\w+)\s*\)'
        r'\s+([\d.]+)\s*/\s*(\d+)',
        re.MULTILINE
    )

    summaries = []
    fin_m = re.search(r'Final result', content)
    if fin_m:
        sum_text = content[fin_m.start():]
        nick_team = {gr.player_a_nick: team_a for gr in game_results}
        nick_team.update({gr.player_b_nick: team_b for gr in game_results})

        for pm in SUM_RE.finditer(sum_text):
            raw_name = pm.group(1)
            pnick    = pm.group(2).strip()
            pscore   = float(pm.group(3))
            pgames   = int(pm.group(4))
            pname    = clean_name(raw_name)   # strip team prefix if present
            pteam    = nick_team.get(pnick, "Unknown")

            if not pname:   # skip empty names (e.g. "Final result:" matched)
                continue

            summaries.append(PlayerMatchSummary(
                tournament_round = t_round,
                match_id         = match_id,
                player_name      = pname,
                player_nick      = pnick,
                team             = pteam,
                total_score      = pscore,
                total_games      = pgames,
                efficiency       = round(pscore / pgames, 4) if pgames else 0.0,
            ))

    match_obj = Match(
        tournament_round = t_round,
        team_a           = team_a,
        team_b           = team_b,
        score_a          = score_a,
        score_b          = score_b,
        match_result     = match_result,
        match_winner     = winner(score_a, score_b),
        num_sub_rounds   = len(sub_rounds_found),
        url              = url,
    )

    return match_obj, game_results, summaries

# ── Crawl all ─────────────────────────────────────────────

def crawl_all(output_dir="."):
    session = requests.Session()
    
    print("Discovering match URLs from website...")
    index_url = f"{BASE}/twbc-2026/matches/pairings"
    try:
        html = fetch(index_url, session)
    except Exception as e:
        print(f"Failed to fetch index page: {e}")
        return
        
    match_urls = {}
    for m in re.finditer(r'href="([^"]*/twbc-2026/matches/round-(\d+)/([^"?]+))"', html):
        link = m.group(1).split("?")[0]   # strip query strings like ?authuser=0
        t_round = int(m.group(2))
        slug = m.group(3)

        if not slug or "round-" in slug:
            continue

        full_url = f"https://sites.google.com{link}" if link.startswith("/") else link
        if t_round not in match_urls:
            match_urls[t_round] = set()
        match_urls[t_round].add(full_url)
        
    match_urls = {k: sorted(list(v)) for k, v in match_urls.items()}
    total_matches = sum(len(v) for v in match_urls.values())
    print(f"Discovered {total_matches} matches across {len(match_urls)} rounds.")

    all_matches, all_games, all_summaries = [], [], []

    for t_round in sorted(match_urls.keys()):
        urls = match_urls[t_round]
        print(f"\n=== TOURNAMENT ROUND {t_round} ===")
        for url in urls:
            slug = url.split("/")[-1]
            print(f"  {slug} ...", end=" ", flush=True)
            try:
                html = fetch(url, session)
                match, games, sums = parse_page(html, t_round, url, session)
                if match:
                    all_matches.append(match)
                    all_games.extend(games)
                    all_summaries.extend(sums)
                    print(f"✓  {match.score_a:.1f}:{match.score_b:.1f} | "
                          f"{len(games)} pairings | {len(sums)} players")
                else:
                    print("✗  parse failed")
                time.sleep(1.2)
            except Exception as e:
                print(f"✗  {e}")

    # ── Save CSVs ─────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    df_m = pd.DataFrame([asdict(x) for x in all_matches])
    df_g = pd.DataFrame([asdict(x) for x in all_games])
    df_s = pd.DataFrame([asdict(x) for x in all_summaries])

    df_m.to_csv(f"{output_dir}/matches.csv",              index=False)
    df_g.to_csv(f"{output_dir}/game_results.csv",         index=False)
    df_s.to_csv(f"{output_dir}/player_match_summary.csv", index=False)

    # ── Report ────────────────────────────────────────────
    print(f"\n=== DONE ===")
    print(f"  matches.csv              : {len(df_m):3d} rows  (expected: 27)")
    print(f"  game_results.csv         : {len(df_g):3d} rows  (expected: ~243)")
    print(f"  player_match_summary.csv : {len(df_s):3d} rows  (expected: ~160)")

    if not df_m.empty:
        print("\n--- matches ---")
        print(df_m[['tournament_round','team_a','team_b',
                     'score_a','score_b','match_winner']].to_string(index=False))

    if not df_g.empty:
        print("\n--- sample game_results ---")
        cols = ['match_id','sub_round','player_a_nick',
                'wins','draws','losses','player_b_nick']
        print(df_g[cols].head(9).to_string(index=False))

    if not df_s.empty:
        print("\n--- top 10 players by efficiency (min 24 games across all matches) ---")
        agg = df_s.groupby('player_nick').agg(
            player_name  = ('player_name', 'first'),
            team         = ('team', 'first'),
            total_score  = ('total_score', 'sum'),
            total_games  = ('total_games', 'sum'),
        )
        agg = agg[agg.total_games >= 24]
        agg['efficiency'] = (agg.total_score / agg.total_games).round(4)
        top = agg.sort_values('efficiency', ascending=False).head(10)
        print(top[['player_name','team','total_score','total_games','efficiency']].to_string())

    return df_m, df_g, df_s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw",
                    help="Output folder (default: data/raw)")
    args = ap.parse_args()
    crawl_all(args.out)