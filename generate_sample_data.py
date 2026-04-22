"""
generate_sample_data.py
=======================
Generates realistic synthetic IPL data (matches.csv + deliveries.csv)
so the dashboard runs without a real Kaggle download.

Usage:
    python generate_sample_data.py

Produces:
    data/matches.csv      (~950 rows — 18 seasons × ~55 matches)
    data/deliveries.csv   (~400k rows — ball-by-ball for every match)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

SEASONS = list(range(2008, 2026))          # 2008–2025
MATCHES_PER_SEASON = 55                    # ~league stage + playoffs

TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

# Teams available per season (some franchises only from certain years)
def available_teams(season: int) -> list:
    base = TEAMS[:8]
    if season >= 2022:
        base = base + TEAMS[8:]          # LSG and GT from 2022
    return base

VENUES = [
    "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium, Bengaluru",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "MA Chidambaram Stadium, Chennai",
    "Rajiv Gandhi International Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali",
    "Narendra Modi Stadium, Ahmedabad",
    "Ekana Cricket Stadium, Lucknow",
]

# Generic player pools
BATTERS = [
    "V Kohli", "RG Sharma", "MS Dhoni", "AB de Villiers", "SR Watson",
    "DA Warner", "KL Rahul", "SK Raina", "G Gambhir", "RV Uthappa",
    "AM Rahane", "SS Iyer", "HH Pandya", "RA Jadeja", "KA Pollard",
    "DJ Bravo", "CH Gayle", "Q de Kock", "JC Buttler", "SE Marsh",
    "SV Samson", "PA Patel", "WP Saha", "BB McCullum", "NV Ojha",
    "KS Williamson", "F du Plessis", "Shubman Gill", "YBK Jaiswal",
    "Tilak Varma", "DP Conway", "RK Singh",
]

BOWLERS = [
    "SL Malinga", "B Kumar", "JJ Bumrah", "YS Chahal", "Harbhajan Singh",
    "PP Chawla", "DJ Bravo", "RA Jadeja", "A Nehra", "RP Singh",
    "IK Pathan", "Z Khan", "AM Mishra", "SP Narine", "AD Russell",
    "MM Sharma", "T Natarajan", "Mohammed Shami", "KV Sharma",
    "Rashid Khan", "Avesh Khan", "A Nortje", "M Morkel", "DL Chahar",
]

ALL_PLAYERS = list(set(BATTERS + BOWLERS))


# ── MATCH GENERATION ───────────────────────────────────────────────────────────

def generate_matches() -> pd.DataFrame:
    rows = []
    match_id = 1
    for season in SEASONS:
        teams = available_teams(season)
        start = date(season, 3, 22)
        for i in range(MATCHES_PER_SEASON):
            t1, t2 = random.sample(teams, 2)
            toss_winner = random.choice([t1, t2])
            toss_decision = random.choice(["bat", "field"])
            venue = random.choice(VENUES)
            match_date = start + timedelta(days=i * (max(1, 60 // MATCHES_PER_SEASON)))

            # ~5 % no-result
            if random.random() < 0.05:
                winner = pd.NA
                result = "no result"
                margin = 0
            else:
                winner = random.choice([t1, t2])
                if random.random() < 0.5:
                    result = "runs"
                    margin = random.randint(1, 90)
                else:
                    result = "wickets"
                    margin = random.randint(1, 10)

            rows.append({
                "id": match_id,
                "season": str(season),
                "city": venue.split(",")[-1].strip(),
                "date": match_date.strftime("%Y-%m-%d"),
                "match_type": "T20",
                "player_of_match": random.choice(ALL_PLAYERS),
                "venue": venue,
                "team1": t1,
                "team2": t2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "result": result,
                "result_margin": margin,
                "target_runs": 0,      # filled after deliveries
                "target_overs": 20,
                "super_over": "N",
                "method": pd.NA,
                "umpire1": "umpire_A",
                "umpire2": "umpire_B",
            })
            match_id += 1

    return pd.DataFrame(rows)


# ── INNINGS SIMULATION ─────────────────────────────────────────────────────────

def _simulate_innings(match_id: int, innings: int,
                      batting_team: str, bowling_team: str,
                      target: int = None) -> list:
    """
    Simulate a T20 innings ball-by-ball.
    Returns list of delivery dicts.
    """
    rows = []
    total_runs = 0
    wickets = 0
    batting_order = random.sample(BATTERS, min(11, len(BATTERS)))
    bowling_order = random.sample(BOWLERS, min(8, len(BOWLERS)))

    batter_idx = 0        # current striker
    non_striker_idx = 1   # non-striker
    bowler_cycle = 0

    for over in range(20):
        bowler = bowling_order[bowler_cycle % len(bowling_order)]
        bowler_cycle += 1
        legal_balls = 0

        while legal_balls < 6:
            batter = batting_order[min(batter_idx, len(batting_order) - 1)]
            non_striker = batting_order[min(non_striker_idx, len(batting_order) - 1)]

            # Early finish: all out or target chased
            if wickets >= 10:
                break
            if target and total_runs >= target:
                break

            # Decide ball outcome
            rand = random.random()
            wide = rand < 0.03
            noball = (not wide) and rand < 0.05
            is_extra = wide or noball

            if wide:
                extras = 1
                runs_off_bat = 0
                player_dismissed = ""
                dismissal_kind = ""
                fielder = ""
                total_runs += 1
                legal_balls -= 1   # wide doesn't count as legal
            elif noball:
                extras = 1
                runs_off_bat = random.choice([0, 1, 2, 4, 6])
                player_dismissed = ""
                dismissal_kind = ""
                fielder = ""
                total_runs += runs_off_bat + 1
            else:
                extras = 0
                r = random.random()
                if r < 0.35:
                    runs_off_bat = 0
                elif r < 0.55:
                    runs_off_bat = 1
                elif r < 0.65:
                    runs_off_bat = 2
                elif r < 0.70:
                    runs_off_bat = 3
                elif r < 0.82:
                    runs_off_bat = 4
                elif r < 0.88:
                    runs_off_bat = 6
                elif r < 0.95:
                    runs_off_bat = 0   # dot + wicket chance below
                else:
                    runs_off_bat = 1

                # Wicket chance (~6 %)
                if random.random() < 0.055 and wickets < 10:
                    player_dismissed = batter
                    dk_options = ["caught", "bowled", "lbw", "run out",
                                  "stumped", "caught and bowled", "hit wicket"]
                    dismissal_kind = random.choice(dk_options)
                    fielder = random.choice(BOWLERS) if "caught" in dismissal_kind else ""
                    wickets += 1
                    batter_idx = max(batter_idx, non_striker_idx) + 1
                    runs_off_bat = 0
                else:
                    player_dismissed = ""
                    dismissal_kind = ""
                    fielder = ""

                total_runs += runs_off_bat

            legal_balls += 1
            ball_in_over = legal_balls   # 1-indexed in raw data

            rows.append({
                "match_id": match_id,
                "innings": innings,
                "over": over,
                "ball": ball_in_over,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "striker": batter,
                "non_striker": non_striker,
                "bowler": bowler,
                "runs_off_bat": runs_off_bat,
                "extras": extras,
                "wides": 1 if wide else 0,
                "noballs": 1 if noball else 0,
                "byes": 0,
                "legbyes": 0,
                "penalty": 0,
                "total_runs": runs_off_bat + extras,
                "player_dismissed": player_dismissed,
                "dismissal_kind": dismissal_kind,
                "fielder": fielder,
            })

            # Rotate strike on odd runs
            if runs_off_bat % 2 == 1:
                batter_idx, non_striker_idx = non_striker_idx, batter_idx

        # Rotate strike at end of over
        batter_idx, non_striker_idx = non_striker_idx, batter_idx

        if wickets >= 10:
            break
        if target and total_runs >= target:
            break

    return rows, total_runs, wickets


# ── DELIVERY GENERATION ────────────────────────────────────────────────────────

def generate_deliveries(matches: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    print(f"Simulating {len(matches)} matches...")

    for idx, match in matches.iterrows():
        mid = match["id"]
        t1, t2 = match["team1"], match["team2"]

        # Who bats first?
        if match["toss_decision"] == "bat":
            bat1, bat2 = match["toss_winner"], (t2 if match["toss_winner"] == t1 else t1)
        else:
            bat2, bat1 = match["toss_winner"], (t2 if match["toss_winner"] == t1 else t1)

        # Innings 1
        inn1_rows, inn1_total, _ = _simulate_innings(mid, 1, bat1, bat2)

        # Innings 2 (chasing)
        target = inn1_total + 1
        inn2_rows, inn2_total, inn2_wkts = _simulate_innings(mid, 2, bat2, bat1, target=target)

        all_rows.extend(inn1_rows)
        all_rows.extend(inn2_rows)

        # Determine actual winner
        if pd.isna(match["winner"]):
            pass   # keep no result
        else:
            if inn2_total >= target:
                actual_winner = bat2
            else:
                actual_winner = bat1
            matches.at[idx, "winner"] = actual_winner
            if inn2_total >= target:
                matches.at[idx, "result"] = "wickets"
                matches.at[idx, "result_margin"] = 10 - inn2_wkts
            else:
                matches.at[idx, "result"] = "runs"
                matches.at[idx, "result_margin"] = inn1_total - inn2_total
            matches.at[idx, "target_runs"] = inn1_total + 1

        if (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(matches)} matches done")

    return pd.DataFrame(all_rows)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  IPL Synthetic Data Generator")
    print("=" * 55)

    print("\n[1/3] Generating match fixtures...")
    matches = generate_matches()
    print(f"      → {len(matches)} matches across {len(SEASONS)} seasons")

    print("\n[2/3] Simulating ball-by-ball deliveries...")
    deliveries = generate_deliveries(matches)

    # Rename striker → batter to match dashboard expectations
    deliveries = deliveries.rename(columns={"striker": "batter"})

    print(f"      → {len(deliveries):,} deliveries generated")

    print("\n[3/3] Writing CSV files...")
    matches_path = os.path.join(OUT_DIR, "matches.csv")
    deliveries_path = os.path.join(OUT_DIR, "deliveries.csv")

    matches.to_csv(matches_path, index=False)
    deliveries.to_csv(deliveries_path, index=False)

    print(f"      ✅  {matches_path}  ({os.path.getsize(matches_path) // 1024} KB)")
    print(f"      ✅  {deliveries_path}  ({os.path.getsize(deliveries_path) // 1024} KB)")

    # Quick sanity checks
    print("\n── Sanity Checks ─────────────────────────────────────")
    print(f"  Seasons       : {sorted(matches['season'].unique().tolist())}")
    print(f"  Teams         : {matches['team1'].nunique()} unique")
    print(f"  Matches       : {len(matches)}")
    print(f"  Deliveries    : {len(deliveries):,}")
    print(f"  Avg inn1 score: {deliveries[deliveries['innings']==1].groupby('match_id')['total_runs'].sum().mean():.1f}")
    print(f"  Wicket rows   : {(deliveries['player_dismissed'] != '').sum():,}")
    print("\n✅  Done! You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
