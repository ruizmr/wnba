# WEHOOP → Internal Data Model – Column Lineage

> This table captures the canonical mapping from raw WEHOOP parquet columns to
> the logical fields exposed by our ingestion layer (`LineRow`, `ResultRow`,
> `ScheduleRow`, `PlayerBoxRow`).  Any lossy transformations (type coercion,
> unit conversion, default-filling) must be called-out explicitly.

| WEHOOP file | Source column | Target model & field | Notes |
|-------------|--------------|----------------------|-------|
| `pbp/play_by_play_*.parquet` | `game_id` | `LineRow.game_id` | copied
|  | `game_date_time` | `LineRow.event_time` | timestamp(US/Eastern) → UTC naïve
|  | `season` | `LineRow.season` | — |
|  | `home_team_id` / `away_team_id` | `LineRow.home_team_id` / `away_team_id` | — |
|  | `home_score` / `away_score` | `LineRow.home_score` / `away_score` | running score at event time |
|  | `home_team_spread` / `game_spread` | `LineRow.home_team_spread` / `game_spread` | NaN allowed |
| `team_box/team_box_*.parquet` | `game_id` | `ResultRow.game_id` | — |
|  | `team_id` | `ResultRow.team_id` | — |
|  | `opponent_team_id` | `ResultRow.opponent_team_id` | derived via merge |
|  | `team_score` / `opponent_team_score` | `ResultRow.team_score` / `opponent_team_score` | — |
|  | `team_winner` | `ResultRow.win_flag` | bool |
| `player_box/player_box_*.parquet` | `athlete_id` | `PlayerBoxRow.athlete_id` | — |
|  | `team_id` | `PlayerBoxRow.team_id` | — |
|  | `minutes` | `PlayerBoxRow.minutes` | float minutes |
|  | `plus_minus` | `PlayerBoxRow.plus_minus` | string in raw → float (coercion) |
| `schedules/wnba_schedule_*.parquet` | `id` | `ScheduleRow.game_id` | — |
|  | `game_date_time` | `ScheduleRow.scheduled_start` | timestamp NY → UTC |
|  | `season` / `season_type` | `ScheduleRow.season`, `season_type` | — |
|  | `home_id` / `away_id` | `ScheduleRow.home_team_id`, `away_team_id` | — |
|  | `status_type_state` | `ScheduleRow.status` | Enum-like string |

### Lossy or Non-trivial Transforms

* `plus_minus` comes through as a **string** in WEHOOP (“+3”, “-8”).  We strip
  the sign and cast to `float` (NaN if empty).
* Percentage columns in `team_box` are stored as **0‒1 floats**; we leave as-is
  but note that upstream ESPN API occasionally emits `null` when a team has no
  attempts.
* Timestamps are normalised to UTC naïve `datetime` throughout the ingestion
  pipeline to avoid downstream timezone surprises.