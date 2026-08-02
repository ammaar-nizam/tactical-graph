# TacticalGraph Neo4j Graph Data Model & Schema Specification

This document details the node labels, relationships, properties, uniqueness constraints, and performance indexes for the TacticalGraph Knowledge Graph.

---

## 1. Node Definitions

| Label | Primary Key / Constraint | Properties |
|---|---|---|
| `:Country` | `id` | `id`, `name`, `code`, `confederation`, `totalClubs`, `totalPlayers`, `averageAge`, `url` |
| `:Competition` | `id` | `id`, `code`, `name`, `subType`, `type`, `domesticLeagueCode`, `confederation`, `totalClubs`, `url` |
| `:NationalTeam` | `id` | `id`, `name`, `teamCode`, `confederation`, `teamImageUrl`, `squadSize`, `averageAge`, `foreignersNumber`, `foreignersPercentage`, `totalMarketValue`, `coachName`, `fifaRanking`, `lastSeason`, `url` |
| `:Club` | `id` | `id`, `code`, `name`, `totalMarketValue`, `squadSize`, `averageAge`, `foreignersNumber`, `foreignersPercentage`, `nationalTeamPlayers`, `stadiumName`, `stadiumSeats`, `netTransferRecord`, `coachName`, `lastSeason`, `fileName`, `url` |
| `:Player` | `id` | `id`, `firstName`, `lastName`, `name`, `lastSeason`, `code`, `countryOfBirth`, `cityOfBirth`, `countryOfCitizenship`, `dateOfBirth`, `subPosition`, `position`, `foot`, `heightInCm`, `contractExpirationDate`, `agentName`, `imageUrl`, `internationalCaps`, `internationalGoals`, `url` |
| `:PlayerValuation` | `id` | `id`, `date`, `marketValueInEur` |
| `:Game` | `id` | `id`, `season`, `round`, `date`, `homeClubGoals`, `awayClubGoals`, `homeClubPosition`, `awayClubPosition`, `homeClubManagerName`, `awayClubManagerName`, `stadium`, `attendance`, `referee`, `url`, `homeClubFormation`, `awayClubFormation`, `aggregate` |
| `:GameEvent` | `id` | `id`, `date`, `minute`, `type`, `description` |
| `:Watermark` | `id` (`'global'`) | `id`, `last_processed_date`, `updated_at` |

---

## 2. Relationship Taxonomy

### 2.1 Structural & Reference Relationships
- `(Competition)-[:LOCATED_IN]->(Country)`
- `(NationalTeam)-[:REPRESENTS_COUNTRY]->(Country)`
- `(Club)-[:COMPETES_IN]->(Competition)`

### 2.2 Player History & Affiliations
- `(Player)-[:PLAYS_FOR]->(Club)` *(Represents active current club)*
- `(Player)-[:REPRESENTS]->(NationalTeam)`
- `(Player)-[:HAS_VALUATION]->(PlayerValuation)`
- `(Player)-[:TRANSFERRED_TO {transferDate, transferSeason, transferFee, transferPeriod, marketValueAtTransfer, fromClubId, fromClubName}]->(Club)`
  - *Note: Points directly to destination club. `fromClubId` is stored as a property for high-performance filtering without multi-hop temporal traversals.*
  - *Data sources: `davidcariboo/player-scores` (ID-based), supplemented by `mexwell/football-player-transfers` (name-based matching).*

### 2.3 Match Containers & Team Performance
- `(Game)-[:PART_OF_COMPETITION]->(Competition)`
- `(Club)-[:PLAYED_IN {hosting, isWin, ownGoals, opponentGoals, ownManagerName, opponentManagerName, ownPosition, opponentPosition}]->(Game)`
  - *Note: `hosting` indicates "Home" or "Away". Captures team match stats and management.*

### 2.4 Player Match Performance
- `(Player)-[:APPEARED_IN {minutesPlayed, type, position, number, teamCaptain, goals, assists, yellowCards, redCards}]->(Game)`
  - *Note: Merges `appearances.csv` and `game_lineups.csv` into a single rich edge payload.*

### 2.5 In-Game Semantic Events
- `(GameEvent)-[:OCCURRED_IN]->(Game)`
- `(Club)-[:INVOLVED_IN]->(GameEvent)`
- `(Player)-[:SCORED]->(GameEvent)` *(Applies when event `type == 'Goals'`)*
- `(Player)-[:SUBBED_OUT]->(GameEvent)` *(Applies when event `type == 'Substitutions'` for outgoing player)*
- `(Player)-[:RECEIVED_CARD]->(GameEvent)` *(Applies when event `type == 'Cards'`)*

---

## 3. Schema Constraints & Indexes

### Uniqueness Constraints
```cypher
CREATE CONSTRAINT country_id_unique IF NOT EXISTS FOR (c:Country) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT competition_id_unique IF NOT EXISTS FOR (c:Competition) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT national_team_id_unique IF NOT EXISTS FOR (n:NationalTeam) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT club_id_unique IF NOT EXISTS FOR (c:Club) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT player_id_unique IF NOT EXISTS FOR (p:Player) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT player_valuation_id_unique IF NOT EXISTS FOR (pv:PlayerValuation) REQUIRE pv.id IS UNIQUE;
CREATE CONSTRAINT game_id_unique IF NOT EXISTS FOR (g:Game) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT game_event_id_unique IF NOT EXISTS FOR (ge:GameEvent) REQUIRE ge.id IS UNIQUE;
```

### Performance Indexes
```cypher
CREATE INDEX player_name_index IF NOT EXISTS FOR (p:Player) ON (p.name);
CREATE INDEX club_name_index IF NOT EXISTS FOR (c:Club) ON (c.name);
CREATE INDEX game_date_index IF NOT EXISTS FOR (g:Game) ON (g.date);
```