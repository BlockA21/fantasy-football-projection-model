import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/weekly_player_stats_offense.csv")
df.columns = df.columns.str.lower()
df = df[df['season'].between(2020, 2024)]  # 5 seasons for context
df['position'] = df['position'].str.upper()

# ----------------------------
# WEEKLY PPR SCORING (ALL POSITIONS)
# ----------------------------
def fantasy_points(row):
    points = 0
    # Passing (QB)
    points += row.get('passing_yards', 0) / 25
    points += row.get('pass_touchdown', 0) * 4
    points -= row.get('interception', 0) * 1
    # Rushing (ALL)
    points += row.get('rushing_yards', 0) / 10
    points += row.get('rush_touchdown', 0) * 6
    # Receiving (RB / WR / TE)
    points += row.get('receptions', 0) * 1
    points += row.get('receiving_yards', 0) / 10
    points += row.get('receiving_touchdown', 0) * 6
    # Fumbles
    points -= row.get('fumble_lost', 0) * 2
    return points

df['weekly_points'] = df.apply(fantasy_points, axis=1)

# ----------------------------
# AGGREGATE PER PLAYER PER SEASON
# ----------------------------
season_totals = df.groupby(
    ['player_id','player_name','position','team','season','age']
)['weekly_points'].sum().reset_index()

# ----------------------------
# FILTER RETIRED / ABSENT PLAYERS
# Must have played in last 3 seasons (2022-2024)
# ----------------------------
active_players = season_totals.groupby('player_id')['season'].max()
active_players = active_players[active_players >= 2022].index
season_totals = season_totals[season_totals['player_id'].isin(active_players)]

# ----------------------------
# SEASON WEIGHTING (last 3 seasons)
# ----------------------------
BASE_WEIGHTS = {2024: 0.7, 2023: 0.2, 2022: 0.1}

def apply_weights(player_df):
    seasons_played = player_df['season'].tolist()
    weights = {s: BASE_WEIGHTS.get(s, 0) for s in seasons_played}
    # Rescale weights if some season missing
    total_weight = sum(weights.values())
    for s in weights:
        weights[s] /= total_weight
    player_df['weight'] = player_df['season'].map(weights)
    player_df['weighted_points'] = player_df['weekly_points'] * player_df['weight']
    return player_df

season_totals = season_totals.groupby('player_id', group_keys=False).apply(apply_weights)

# ----------------------------
# INJURY / MISSING SEASON ADJUSTMENT
# Players missing 2024 get 10% deduction
# ----------------------------
last_season = season_totals.groupby('player_id')['season'].max().reset_index()
last_season['injury_penalty'] = np.where(last_season['season'] < 2024, 0.9, 1.0)
season_totals = season_totals.merge(last_season[['player_id','injury_penalty']], on='player_id', how='left')

# ----------------------------
# AGGREGATE FINAL PLAYER PROJECTIONS
# ----------------------------
players = season_totals.groupby(
    ['player_id','player_name','position','team','age','injury_penalty']
).agg(
    base_projection=('weighted_points','sum'),
    consistency=('weekly_points','std')
).reset_index()

# ----------------------------
# AGE DECAY BY POSITION
# ----------------------------
def age_decay(row):
    age, pos = row['age'], row['position']
    factor = 1.0
    if pos == 'QB':
        if age >= 33: factor *= 0.95
        if age >= 35: factor *= 0.90
    elif pos == 'RB':
        if age >= 28: factor *= 0.95
        if age >= 30: factor *= 0.90
    elif pos == 'WR':
        if age >= 31: factor *= 0.95
        if age >= 33: factor *= 0.90
    elif pos == 'TE':
        if age >= 32: factor *= 0.95
    return factor

players['age_factor'] = players.apply(age_decay, axis=1)

# ----------------------------
# CONSISTENCY SCORE
# ----------------------------
players['consistency_score'] = 1 / (1 + players['consistency'].fillna(0))

# ----------------------------
# LATE-SEASON BOOST (last 7 games of 2024)
# ----------------------------
late_2024 = df[df['season']==2024].sort_values(['player_id','week'])
late_2024 = late_2024.groupby('player_id').tail(7)
late_points = late_2024.groupby('player_id')['weekly_points'].sum().reset_index()
late_points.rename(columns={'weekly_points':'late_boost'}, inplace=True)

# Late boost by position
def apply_late_boost(row):
    if row['position'] == 'QB':
        return row['late_boost'] * 0.03
    else:  # RB/WR/TE
        return row['late_boost'] * 0.045

players = players.merge(late_points, on='player_id', how='left')
players['late_boost'] = players['late_boost'].fillna(0)
players['late_boost'] = players.apply(apply_late_boost, axis=1)

# ----------------------------
# CAREER-ELITE BOOST (top 20% per position across 2020-2024)
# ----------------------------
career_totals = df.groupby(['player_id','position'])['weekly_points'].sum().reset_index()
career_elite = career_totals.groupby('position')['weekly_points'].quantile(0.8).reset_index()
career_elite.rename(columns={'weekly_points':'elite_threshold'}, inplace=True)
career_totals = career_totals.merge(career_elite, on='position', how='left')
career_totals['elite_boost'] = np.where(career_totals['weekly_points'] >= career_totals['elite_threshold'], 1.05, 1.0)

players = players.merge(career_totals[['player_id','elite_boost']], on='player_id', how='left')
players['elite_boost'] = players['elite_boost'].fillna(1.0)

# ----------------------------
# FINAL PROJECTION
# ----------------------------
players['projected_2025'] = (
    players['base_projection']
    * players['age_factor']
    * players['injury_penalty']
    * players['consistency_score']
    * players['elite_boost']
    + players['late_boost']
)

# ----------------------------
# TOP PLAYERS BY POSITION
# ----------------------------
def top_n(pos, n):
    return players[players['position']==pos].sort_values('projected_2025', ascending=False).head(n)

top_qb = top_n('QB', 10)
top_rb = top_n('RB', 15)
top_wr = top_n('WR', 15)
top_te = top_n('TE', 8)

final = pd.concat([top_qb, top_rb, top_wr, top_te])
final.to_csv("data/projected_2025_fantasy_rankings.csv", index=False)

# ----------------------------
# VISUALIZATION
# ----------------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(4,1, figsize=(14,18))

for ax, pos, n in zip(axes, ['QB','RB','WR','TE'], [10,15,15,8]):
    data = final[final['position']==pos].sort_values('projected_2025')
    sns.barplot(x='projected_2025', y='player_name', data=data, ax=ax)
    ax.set_title(f"Top {n} {pos} – Projected 2025 Fantasy PPR")

plt.tight_layout()
plt.show()

print(final[['player_name','position','team','age','projected_2025']])
