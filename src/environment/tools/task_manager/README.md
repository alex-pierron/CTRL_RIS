# TaskManager: logique détaillée et modes de difficulté

Ce dossier isole toute la logique curriculum autour de `TaskManager`.

## Vue d’ensemble

Le `TaskManager` orchestre 4 responsabilités:

1. **Définir les niveaux de difficulté** (templates statiques).
2. **Échantillonner un niveau par environnement** à chaque épisode.
3. **Construire les payloads d’épisode** passés aux `reset` d’environnements.
4. **Faire évoluer le niveau maximal atteignable** selon les outcomes.

## Organisation des fichiers

- `types.py`:
  - `DifficultyLevel`
  - `Outcome`
  - `DifficultyConfig` (template statique)
  - `EpisodeDifficultyConfig` (payload dynamique d’épisode)
- `default_difficulty_modes.py`:
  - `create_default_difficulty_configs(...)`
  - `validate_difficulty_configs(...)`
- `custom_difficulty_modes_template.py`:
  - exemple didactique modifiable par un usager externe
- `core.py`:
  - classe `TaskManager` (sampling, outcomes, progression)

## Contrat de données

### 1) Template de niveau (`DifficultyConfig`)

Objet statique attaché à un niveau (`LEVEL_1` à `LEVEL_5`), contenant:

- `grid_limits`: rectangle de génération `[[x_min, x_max], [y_min, y_max]]`
- `angle_is_max`: drapeau sémantique de contrainte angulaire
- `angle_value`: valeur de contrainte angulaire (radians)
- `fully_random`: active la génération utilisateurs totalement aléatoire
- `new_min_distance_between_eavesdropper_and_users`
- `new_max_distance_between_eavesdropper_and_users`

### 2) Payload épisode (`EpisodeDifficultyConfig`)

Objet construit à partir d’un template puis transmis au `reset`:

- `grid_limits`
- `angle_is_max`
- `angle_value`
- `min_distance_eavesdropper_users`
- `max_distance_eavesdropper_users`
- `fully_random`

Il peut être reconstruit depuis:
- dataclass explicite,
- dictionnaire,
- tuple/list legacy de longueur 6,
- objet duck-typed.

## Logique de sampling curriculum

`TaskManager` maintient `current_max_level` (niveau le plus dur autorisé).

La distribution de sampling dépend de ce niveau:

- `n=1`: 100% niveau 1.
- `n=2`: 80% niveau 2, 20% niveau 1.
- `n>=3`:
  - 65% niveau `n`,
  - 20% niveau `n-1`,
  - 15% répartis uniformément sur `1..n-2`.

À chaque épisode:

1. `generate_episode_configs()` échantillonne `num_environments` niveaux.
2. Chaque niveau est converti en `EpisodeDifficultyConfig`.
3. La liste est passée telle quelle à `VecEnv.reset(...)`.

## Logique outcomes et progression

Entrées possibles:

- `downlink_sum`
- `uplink_sum`
- `best_eavesdropper_sum`

Ces sommes sont normalisées par `num_steps_per_episode`.

Seuils utilisés:

- `thresholds[0]`: seuil downlink.
- `thresholds[1]`: seuil uplink.
- `eavesdropper_thresholds=[eav_dl, eav_ul]` (optionnel): seuils dédiés à la
  condition eavesdropper. Si omis, fallback rétrocompatible:
  - `thresholds=[dl, ul, eav]` -> `[eav, eav]`
  - `thresholds=[dl, ul, eav_dl, eav_ul]` -> `[eav_dl, eav_ul]`
  - sinon réutilise `[dl, ul]`

Par condition:

- **downlink/uplink**: succès si la métrique moyenne est `>` seuil.
- **eavesdropper**: succès si la métrique moyenne est `<` seuil.

Chaque condition produit `SUCCESS`, `FAILURE` ou `SEVERE_FAILURE` selon la
proportion d’utilisateurs satisfaits:

- `ratio == 1.0` => `SUCCESS`
- `ratio >= 0.51` => `FAILURE`
- sinon => `SEVERE_FAILURE`

L’outcome final d’un épisode est le maximum de sévérité entre conditions.

## Règles de progression/régression

Le buffer `episode_buffer` (FIFO) accumule les outcomes récents.

### Progression (niveau +1)

Si buffer plein et:
- `success_rate > 0.9`
- `severe_failure_rate < 0.005`

### Régression (niveau -1)

Si assez d’épisodes ont été joués sur le niveau courant:
- `episodes_used_current_level >= H * Buffer_Size`

et:
- `success_rate < 0.4`

Après progression/régression: reset du buffer et des compteurs de niveau.

## Modes de difficulté par défaut

Les valeurs par défaut sont dans `default_difficulty_modes.py`:

- `LEVEL_1`: zone restreinte, séparation angulaire large, eavesdroppers loin.
- `LEVEL_2`: zone élargie, contrainte angulaire plus stricte.
- `LEVEL_3`: contrainte angulaire encore plus stricte.
- `LEVEL_4`: densité accrue autour des utilisateurs.
- `LEVEL_5`: positionnement utilisateur totalement aléatoire (`fully_random`).

## Personnalisation utilisateur

Utiliser `custom_difficulty_modes_template.py` comme base:

1. Modifier les valeurs niveau par niveau.
2. Conserver les 5 niveaux.
3. Valider avec `validate_difficulty_configs(...)`.
4. Passer le dictionnaire à:
   `TaskManager(..., difficulty_configs=my_custom_configs)`.

## Invariants importants à préserver

- Les 5 clés `DifficultyLevel` doivent toujours exister.
- `grid_limits` doit être de forme `(2, 2)` avec bornes strictement croissantes.
- Distances eavesdropper: `0 <= min <= max`.
- `generate_episode_configs()` et `update_episode_outcomes(...)` doivent traiter
  le même nombre d’environnements.
