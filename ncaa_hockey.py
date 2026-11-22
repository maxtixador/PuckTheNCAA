"""NCAA men's Division I hockey scraping and play-text parsing functions.

Dependencies: numpy, pandas, and requests.

Main functions:
- getBoxScore: Fetch and parse the box score for a given NCAA hockey game ID.
- getPlays: Fetch and parse the play-by-play data for a given NCAA hockey game ID.
- getGamesInRange: Fetch NCAA games in an inclusive date range in MM/DD/YYYY format.
- getPlaysInRange: Fetch play-by-play data for NCAA hockey games in an inclusive date range in MM/DD/YYYY format.

Author: Max
Date: 2026-11-22
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests


def getBoxScore(gameId):
    """
    Fetch and parse the box score for a given NCAA hockey game ID.

    Returns a DataFrame containing the scoring summary for the specified game.
    Raises a ValueError if new goal types are found that are not in the predefined list.
    """
    summary_scoring_url = "https://sdataprod.ncaa.com/?meta=NCAA_GetGamecenterScoringSummaryById_web&extensions={%22persistedQuery%22:{%22version%22:1,%22sha256Hash%22:%22fcd5729c72b0f72a4f659bf07e7b1da0fdce8f41ad286b0ddfe830adc7a45ca3%22}}&variables={%22contestId%22:%22" + str(gameId) + "%22,%22staticTestEnv%22:null}"
    summary_scoring_response = requests.get(summary_scoring_url)
    summary_scoring_data = summary_scoring_response.json()

    scoring_summary = summary_scoring_data.get("data").get('scoringSummary')
    teams, periods = scoring_summary.get('teams'), scoring_summary.get('periods')
    teams_df = pd.DataFrame(teams)
    periods_df = pd.json_normalize(
        periods,
        record_path="summary",
        meta=["title"],
        meta_prefix="period_"
    )
    periods_df = periods_df.merge(teams_df[["teamId", "isHome", "seoname", "name6Char", "nameShort"]], on="teamId")
    periods_df["gameId"] = gameId
    periods_df["meta_url"] = summary_scoring_url
    periods_df["period"] = pd.to_numeric(periods_df["period_title"].str.extract(r"(\d+)[A-Za-z]{2}")[0])
    periods_df["goalScorer"] = periods_df["scoreText"].str.extract(r"^(.*?)\s*\(")[0]
    periods_df["goalA1"] = periods_df["scoreText"].str.extract(r"\(Assist by\s+(.*?)(?:\s+and|\))")[0].fillna("")
    periods_df["goalA2"] = periods_df["scoreText"].str.extract(r"\s+and\s+(.*?)\)")[0].fillna("")
    periods_df["elapsedTime"] = (periods_df["period"] - 1)*60*20 + pd.to_numeric(periods_df["time"].str.extract(r"(\d+):(\d+)").apply(lambda x: int(x[0])*60 + int(x[1]), axis=1))

    score_type_list = ["FG", "PP", "SH", "EN", "EV", "GW", "SD", "UA"]
    goal_types_found = periods_df["scoreType"].apply(lambda x: [gt for gt in score_type_list if gt in x] if pd.notnull(x) else []).explode().dropna().unique()
    new_score_types = [gt for gt in goal_types_found if gt not in score_type_list]
    if new_score_types:
        raise ValueError(f"New goal types found: {new_score_types}")

    periods_df["isFG"] = periods_df["scoreType"].str.contains(r"FG").fillna(False)
    periods_df["isPP"] = periods_df["scoreType"].str.contains(r"PP").fillna(False)
    periods_df["isSH"] = periods_df["scoreType"].str.contains(r"SH").fillna(False)
    periods_df["isEN"] = periods_df["scoreType"].str.contains(r"EN").fillna(False)
    periods_df["isEV"] = periods_df["scoreType"].str.contains(r"EV").fillna(False)
    periods_df["isGW"] = periods_df["scoreType"].str.contains(r"GW").fillna(False)
    periods_df["isSD"] = periods_df["scoreType"].str.contains(r"SD").fillna(False)
    periods_df["isUA"] = periods_df["scoreType"].str.contains(r"UA").fillna(False)
    return periods_df.drop(columns=["__typename"])


def getPlays(gameId=None):
    """Fetch and parse the play-by-play data for a given NCAA hockey game ID."""
    url = "https://sdataprod.ncaa.com/?meta=NCAA_GetGamecenterPbpGenericById_web&extensions={%22persistedQuery%22:{%22version%22:1,%22sha256Hash%22:%2257f922d56d60d88326b62202b3d88e8cd3cfb6687931bc0b5b3dfab089b84faa%22}}&variables={%22contestId%22:%22" + str(gameId) + "%22,%22staticTestEnv%22:null}"
    response = requests.get(url)
    res = response.json()

    data = res.get("data")
    pbp = data.get("playbyplay")
    periods = pbp["periods"]

    df = pd.json_normalize(
        periods,
        record_path=["playbyplayStats", "plays"],
        meta=[
            "periodNumber",
            "periodDisplay",
            ["playbyplayStats", "clock"],
            ["playbyplayStats", "teamId"],
        ],
        errors="ignore"
    )

    txt = df['playText'].str.strip()
    low = txt.str.lower()
    df['event_type'] = np.select(
        [low.str.startswith('goal by'),
         low.str.startswith('shot by'),
         low.str.startswith('penalty on'),
         low.str.contains('at goalie for', regex=False),
         low.str.startswith(('end power play', 'start power play')),
         low.str.contains('media time out', regex=False)],
        ['Goal', 'Shot', 'Penalty', 'Goalie Change', 'Power Play Status', 'Media Timeout'],
        default='Other')

    PATTERNS = {
        'Shot': (r'^Shot by (?P<shot_team>\S+) (?P<shot_player>.+?) '
                 r'(?P<shot_result>MISSED|BLOCKED|WIDE|HIT PIPE|HIT CROSSBAR)'
                 r'(?:, save (?P<goalie>.+?))?(?: by (?P<blocker>.+?))?\.?$'),
        'Goal': (r'^Goal by (?P<scorer>[^(]+?)\s*\((?P<strength>[^)]+)\)'
                 r'(?:\s*Assist by:\s*(?P<assist1>[^,]+?)(?:,\s*(?P<assist2>[^,]+?))?)?'
                 r',?\s*On ice for (?P<on_ice_team1>\w+):\s*(?P<on_ice_players1>.+?)'
                 r'(?:\s+(?P<on_ice_team2>\w+):\s*(?P<on_ice_players2>.+))?$'),
        'Penalty': (r'^Penalty on (?P<pen_player>.+?) (?P<pen_team>\S+) '
                    r'(?P<pen_minutes>\d+) minutes? for (?P<pen_infraction>.+?)\.?$'),
        'Goalie Change': r'^(?P<goalie_name>.+?) at goalie for (?P<goalie_team>.+)$',
    }

    parts = []
    for ev, pat in PATTERNS.items():
        mask = df['event_type'].eq(ev)
        parts.append(txt[mask].str.extract(pat, flags=re.I | re.S))

    extracted = pd.concat(parts)
    extracted = extracted.apply(lambda c: c.str.strip() if c.dtype == object else c)
    extracted['shot_result'] = extracted['shot_result'].str.upper()
    extracted['pen_infraction'] = extracted['pen_infraction'].str.lower()
    extracted['pen_minutes'] = pd.to_numeric(extracted['pen_minutes'])
    parsed = extracted.groupby(level=0).first()
    df[parsed.columns] = parsed

    df["gameId"] = pbp["contestId"]
    df["gameDescription"] = pbp["description"]
    df["divisionName"] = pbp["divisionName"]
    df["gameStatus"] = pbp["status"]

    for team in pbp["teams"]:
        if team["isHome"]:
            df["homeTeamId"] = team["teamId"]
            df["homeSeoName"] = team["seoname"]
            df["homeName6Char"] = team["name6Char"]
            df["homeNameShort"] = team["nameShort"]
            df["homeNameFull"] = team["nameFull"]
            df["homeTeamName"] = team["teamName"]
            df["homeColor"] = team["color"]
        else:
            df["awayTeamId"] = team["teamId"]
            df["awaySeoName"] = team["seoname"]
            df["awayName6Char"] = team["name6Char"]
            df["awayNameShort"] = team["nameShort"]
            df["awayNameFull"] = team["nameFull"]
            df["awayTeamName"] = team["teamName"]
            df["awayColor"] = team["color"]

    return df


def getGames(month, day, year, raise_error=True):
    """Fetch NCAA games for a specific date (month, day, year)."""
    url = "https://sdataprod.ncaa.com/?meta=GetContests_web&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%224bcb5e6432fa9da365c0c19af01b1f9015cc7eb5c21e7af2dba308784a166df7%22%7D%7D&queryName=GetContests_web&variables=%7B%22sportCode%22%3A%22MIH%22%2C%22division%22%3A1%2C%22seasonYear%22%3A2026%2C%22contestDate%22%3A%22" + str(month) + "%2F" + str(day) + "%2F" + str(year) + "%22%2C%22week%22%3Anull%7D"

    response = requests.get(url)
    res = response.json()
    data = res.get("data")
    contests = data.get("contests")

    if not contests:
        if raise_error:
            raise ValueError("No contests found for the given date.")
        return pd.DataFrame()

    df = pd.json_normalize(contests)
    home = pd.json_normalize(df["teams"].str[0])
    away = pd.json_normalize(df["teams"].str[1])
    return pd.concat([
        df.drop(columns=["teams", "liveVideos", "__typename"]),
        home.add_prefix("home_"),
        away.add_prefix("away_")
    ], axis=1)


def getGamesInRange(start_date, end_date):
    """
    Fetch NCAA games in an inclusive date range in MM/DD/YYYY format.
    
    Returns a DataFrame containing all NCAA games within the specified date range.
    If no games are found, returns an empty DataFrame.
    
    Uses ThreadPoolExecutor for concurrent fetching of games.
    """
    start_month, start_day, start_year = map(int, start_date.split("/"))
    end_month, end_day, end_year = map(int, end_date.split("/"))
    current_date = date(start_year, start_month, start_day)
    end_date = date(end_year, end_month, end_day)
    all_games = []
    dates = []
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(getGames, d.month, d.day, d.year, raise_error=False): d for d in dates}
        for future in as_completed(futures):
            games = future.result()
            if not games.empty:
                all_games.append(games)

    if all_games:
        return pd.concat(all_games, ignore_index=True)
    return pd.DataFrame()

def getPlaysInRange(start_date, end_date):
    """
    Fetch play-by-play data for NCAA hockey games in an inclusive date range in MM/DD/YYYY format.
    
    Returns a DataFrame containing all play-by-play events within the specified date range.
    If no plays are found, returns an empty DataFrame.
    
    Uses ThreadPoolExecutor for concurrent fetching of schedule (through getGamesInRange) and play-by-play data.
    """
    games = getGamesInRange(start_date, end_date)
    all_plays = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(getPlays, gameId): gameId for gameId in games["gameId"].unique()}
        for future in as_completed(futures):
            plays = future.result()
            if not plays.empty:
                all_plays.append(plays)
    if all_plays:
        return pd.concat(all_plays, ignore_index=True)
    return pd.DataFrame()


def categorize(text):
    t = text.lower()
    if t.startswith('goal by'):
        return 'Goal'
    if t.startswith('shot by'):
        return 'Shot'
    if t.startswith('penalty on'):
        return 'Penalty'
    if 'at goalie for' in t:
        return 'Goalie Change'
    if t.startswith('end power play') or t.startswith('start power play'):
        return 'Power Play Status'
    if 'media time out' in t:
        return 'Media Timeout'
    return 'Other'


shot_pat = re.compile(
    r'^Shot by (?P<team>[\w\s\.]+?) (?P<player>[^,]+?) '
    r'(?P<result>MISSED|BLOCKED|WIDE|HIT PIPE|HIT CROSSBAR)'
    r'(?:, save (?P<goalie>[^,]+))?'
    r'(?: by (?P<blocker>.+))?\.?$',
    re.IGNORECASE
)


def extract_shot(text):
    m = shot_pat.match(text)
    if not m:
        return pd.Series({
            'shot_team': np.nan, 'shot_player': np.nan,
            'shot_result': np.nan, 'goalie': np.nan, 'blocker': np.nan
        })
    d = m.groupdict()
    return pd.Series({
        'shot_team': d['team'].strip(),
        'shot_player': d['player'].strip(),
        'shot_result': d['result'].upper(),
        'goalie': d.get('goalie'),
        'blocker': d.get('blocker')
    })


goal_pat = re.compile(
    r'^Goal by (?P<scorer>[^(]+?)\s*\((?P<strength>[^)]+)\)'
    r'(?:\s*Assist by:\s*(?P<assists>[^,]+(?:,\s*[^,]+)*))?'
    r'(?:,?\s*On ice for (?P<team1>\w+):\s*(?P<on_ice1>.+?))'
    r'(?:\s+(?P<team2>\w+):\s*(?P<on_ice2>.+))?$',
    re.IGNORECASE | re.DOTALL
)


def extract_goal(text):
    m = goal_pat.search(text)
    if not m:
        return pd.Series(dtype=object)

    d = m.groupdict()
    assists = [a.strip() for a in d['assists'].split(',')] if d['assists'] else []
    return pd.Series({
        'scorer': d['scorer'].strip(),
        'strength': d['strength'].strip(),
        'assist1': assists[0] if len(assists) > 0 else np.nan,
        'assist2': assists[1] if len(assists) > 1 else np.nan,
        'on_ice_team1': d.get('team1'),
        'on_ice_players1': d.get('on_ice1'),
        'on_ice_team2': d.get('team2'),
        'on_ice_players2': d.get('on_ice2'),
    })


pen_pat = re.compile(
    r'^Penalty on (?P<player>[^,]+?) (?P<team>[\w\s\.]+?) '
    r'(?P<minutes>\d+) minutes? for (?P<infraction>.+?)\.?$',
    re.IGNORECASE
)


def extract_penalty(text):
    m = pen_pat.match(text)
    if not m:
        return pd.Series(dtype=object)
    d = m.groupdict()
    return pd.Series({
        'pen_player': d['player'].strip(),
        'pen_team': d['team'].strip(),
        'pen_minutes': int(d['minutes']),
        'pen_infraction': d['infraction'].strip().lower()
    })


goalie_pat = re.compile(
    r'^(?P<goalie>.+?) at goalie for (?P<team>.+)$',
    re.IGNORECASE
)


def extract_goalie(text):
    m = goalie_pat.match(text)
    if not m:
        return pd.Series(dtype=object)
    return pd.Series({
        'goalie_name': m.group('goalie').strip(),
        'goalie_team': m.group('team').strip()
    })
