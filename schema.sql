-- TWBC 2026 — Fully Normalized Schema (BCNF)
-- Run: sqlite3 twbc.db < schema.sql

PRAGMA foreign_keys = ON;

-- ── Stored Tables (source facts only) ────────────────────

CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    nick        TEXT    NOT NULL UNIQUE,
    full_name   TEXT    NOT NULL,
    team_id     INTEGER NOT NULL REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_round INTEGER NOT NULL,
    team_a_id        INTEGER NOT NULL REFERENCES teams(team_id),
    team_b_id        INTEGER NOT NULL REFERENCES teams(team_id),
    match_result     TEXT    NOT NULL,
    source_url       TEXT,
    UNIQUE(tournament_round, team_a_id, team_b_id),
    CHECK(team_a_id != team_b_id)
);

CREATE TABLE IF NOT EXISTS sub_rounds (
    sub_round_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id      INTEGER NOT NULL REFERENCES matches(match_id),
    round_number  INTEGER NOT NULL,
    UNIQUE(match_id, round_number)
);

CREATE TABLE IF NOT EXISTS pairings (
    pairing_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    sub_round_id  INTEGER NOT NULL REFERENCES sub_rounds(sub_round_id),
    player_a_id   INTEGER NOT NULL REFERENCES players(player_id),
    player_b_id   INTEGER NOT NULL REFERENCES players(player_id),
    score_a       REAL    NOT NULL,
    score_b       REAL    NOT NULL,
    UNIQUE(sub_round_id, player_a_id, player_b_id),
    CHECK(player_a_id != player_b_id),
    CHECK(score_a >= 0),
    CHECK(score_b >= 0)
);

-- ── Computed Views ───────────────────────────────────────

CREATE VIEW IF NOT EXISTS v_pairings AS
SELECT
    p.*,
    p.score_a + p.score_b AS total_games,
    CASE
        WHEN p.score_a > p.score_b THEN 'A'
        WHEN p.score_b > p.score_a THEN 'B'
        ELSE 'Draw'
    END AS winner
FROM pairings p;


CREATE VIEW IF NOT EXISTS v_sub_rounds AS
SELECT
    sr.sub_round_id,
    sr.match_id,
    sr.round_number,
    COALESCE(SUM(p.score_a), 0) AS score_a,
    COALESCE(SUM(p.score_b), 0) AS score_b,
    CASE
        WHEN SUM(p.score_a) > SUM(p.score_b) THEN 'A'
        WHEN SUM(p.score_b) > SUM(p.score_a) THEN 'B'
        ELSE 'Draw'
    END AS winner
FROM sub_rounds sr
LEFT JOIN pairings p ON p.sub_round_id = sr.sub_round_id
GROUP BY sr.sub_round_id;


CREATE VIEW IF NOT EXISTS v_matches AS
SELECT
    m.match_id,
    m.tournament_round,
    ta.name             AS team_a,
    tb.name             AS team_b,
    m.team_a_id,
    m.team_b_id,
    COALESCE(agg.score_a, 0) AS score_a,
    COALESCE(agg.score_b, 0) AS score_b,
    m.match_result,
    CASE
        WHEN agg.score_a > agg.score_b THEN 'A'
        WHEN agg.score_b > agg.score_a THEN 'B'
        ELSE 'Draw'
    END AS winner,
    COALESCE(agg.num_sub_rounds, 0) AS num_sub_rounds,
    m.source_url
FROM matches m
JOIN teams ta ON ta.team_id = m.team_a_id
JOIN teams tb ON tb.team_id = m.team_b_id
LEFT JOIN (
    SELECT
        sr.match_id,
        SUM(p.score_a) AS score_a,
        SUM(p.score_b) AS score_b,
        COUNT(DISTINCT sr.sub_round_id) AS num_sub_rounds
    FROM sub_rounds sr
    JOIN pairings p ON p.sub_round_id = sr.sub_round_id
    GROUP BY sr.match_id
) agg ON agg.match_id = m.match_id;


CREATE VIEW IF NOT EXISTS v_player_match_summary AS
SELECT
    m.tournament_round,
    m.match_id,
    p.full_name                   AS player_name,
    p.nick                        AS player_nick,
    t.name                        AS team,
    SUM(s.pts)                    AS total_score,
    SUM(s.gms)                    AS total_games,
    ROUND(SUM(s.pts) * 1.0 / SUM(s.gms), 4) AS efficiency
FROM (
    SELECT sub_round_id, player_a_id AS pid, score_a AS pts, score_a + score_b AS gms
    FROM pairings
    UNION ALL
    SELECT sub_round_id, player_b_id AS pid, score_b AS pts, score_a + score_b AS gms
    FROM pairings
) s
JOIN sub_rounds sr ON sr.sub_round_id = s.sub_round_id
JOIN matches m     ON m.match_id = sr.match_id
JOIN players p     ON p.player_id = s.pid
JOIN teams t       ON t.team_id = p.team_id
GROUP BY m.match_id, p.player_id;


CREATE VIEW IF NOT EXISTS v_player_overall AS
SELECT
    player_nick,
    player_name,
    team,
    SUM(total_score)  AS total_score,
    SUM(total_games)  AS total_games,
    ROUND(SUM(total_score) * 1.0 / SUM(total_games), 4) AS efficiency
FROM v_player_match_summary
GROUP BY player_nick;
