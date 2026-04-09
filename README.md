Camborne School of Mines University of Exeter
# Reinforcement-Learning-mine-scheduler-with-stockpiling
A two-agent (plant mine) reinforcement learning scheduler for open-pit mine planning with stockpiling and plant feed control is presented. The system schedules mining at the cluster rather than at the single-block level, while enforcing operational constraints: precedence, mining capacity, minimum mining width, and exposed-height limits. 

# Reinforcement Learning Based Mine Scheduling with Stockpiling

This repository contains a two-agent reinforcement learning scheduler for open-pit mine planning with stockpiling and plant feed control. The system schedules mining at the **cluster/panel level** rather than at the single-block level, while enforcing operational constraints such as precedence, annual mining capacity, minimum mining width, and optional exposed-height limits. A plant agent chooses yearly plant operating modes, and a mining agent selects feasible mining clusters within each year.

The workflow is driven by a plain-text configuration file (`.txt`) in `key=value` format. Users provide the path to the input block model CSV, the output Excel file, and all economic, mining, plant, slope, and stockpile parameters through this config file.

---

## 1. Main capabilities

- Reads a block model CSV from a path defined in the config file
- Computes tonnage and block value from geometry, cutoff, density, costs, price, and recovery
- Optionally augments precedence using directional slope angles
- Aggregates blocks into **min-width clusters/panels** for practical scheduling actions
- Uses a **two-agent RL formulation**:
  - **Plant agent** selects yearly plant operating mode
  - **Mining agent** selects feasible clusters during the year
- Includes **stockpiling and reclaiming** within plant blending
- Enforces hard operational constraints during scheduling
- Exports schedule and performance summaries to Excel

---

## 2. Repository inputs

### Required input file
A CSV block model is expected at the path given by:

```txt
input_file_name_path=...
```

### Required block model columns
The scheduler expects these columns in the input CSV:

- `Indices`
- `X`
- `Y`
- `Z`
- `Grade`
- `x_axis_len`
- `y_axis_len`
- `z_axis_len`

Strongly recommended optional columns:

- `precedence index`
- `adjacent index`

If `precedence index` or `adjacent index` are missing, the script will initialize them as empty lists.

---

## 3. Configuration file

All runtime options are set in a text file such as `New_txt_file.txt`.

Example:

```txt
input_file_name_path=C:/path/to/block_model.csv
output_file_name_path=C:/path/to/results.xlsx
split_char=,
price=9000
refining_cost=3
mineral_density=2.7
waste_density=2.3
cutoff=0.28
recovery=95
dis_rate=10
p_cost=10
mining_cost=1
rehandling_cost=1
slopes=[35,35,35,35]
use_slope_precedence=1
slope_z_window_blocks=3
mining_capacity=3000000
process_capacity=1000000
minimum_mining_width_defined=40
max_allowable_depth=3
max_allowable_depth_priority_indicator=0
user_define_low_limit=0.8
user_define_up_limit=1.3
numb_of_stockpiles=3
stockpiles=[(0.28,0.8),(0.8,2.5),(2.5,10)]
stockpiles_capacity=[4500000,4500000,4500000]
blender_mode="rl_pure"
solver_selection=1
Condition=0
```

### Important config groups

#### File paths
- `input_file_name_path` : path to the input CSV block model
- `output_file_name_path` : path to the output Excel workbook
- `split_char` : CSV separator

#### Economics
- `price`
- `refining_cost`
- `recovery`
- `p_cost`
- `mining_cost`
- `rehandling_cost`
- `dis_rate`

#### Physical and mining parameters
- `mineral_density`
- `waste_density`
- `cutoff`
- `mining_capacity`
- `process_capacity`
- `minimum_mining_width_defined`
- `Condition`

#### Geotechnical / geometric controls
- `use_slope_precedence`
- `slopes=[N,E,S,W]`
- `slope_z_window_blocks`
- `max_allowable_depth`
- `max_allowable_depth_priority_indicator`

#### Plant and stockpiles
- `user_define_low_limit`
- `user_define_up_limit`
- `numb_of_stockpiles`
- `stockpiles`
- `stockpiles_capacity`
- `blender_mode`

---

## 4. Algorithm overview

### Step 1: Read config and input block model
The script reads the `.txt` config file, parses values using Python-literal parsing where possible, and then loads the block model CSV.

### Step 2: Preprocess blocks
The scheduler:
- validates required columns
- parses precedence and adjacency lists
- computes block tonnage from block dimensions and density
- computes block economic value from grade, recovery, price, mining cost, and processing cost
- sorts blocks from top to bottom for scheduling

### Step 3: Build slope-based precedence (optional)
If enabled, the script computes additional precedence relationships using directional slope angles `[N, E, S, W]` and a vertical search window.

### Step 4: Build min-width clusters (panel actions)
Instead of scheduling one block at a time, the script groups blocks into compact elevation-wise tiles using the minimum mining width. Small or sparse tiles are merged grade-aware to create more practical mining panels.

### Step 5: Build cluster graphs
Block-level precedence and adjacency are converted into cluster-level precedence and neighbour relationships.

### Step 6: Create the two-agent environment
The environment contains:
- cluster inventory and cluster features
- feasible-action masking
- stockpile system and deterministic blender
- current year, remaining mining capacity, plant target grade, reclaim bias, and audit logs

### Step 7: Train the RL scheduler
Two learning agents are trained:

#### Plant agent
At the start of each year, the plant agent selects a **plant mode**. Each mode defines:
- target grade position within the plant window
- reclaim bias (reclaim-heavy, neutral, or mine-heavy)

#### Mining agent
During the year, the mining agent repeatedly selects feasible mining clusters. The environment blocks invalid actions and can auto-fill extra feasible clusters to better pack the year toward mining capacity.

### Step 8: Commit each year
When no more feasible clusters fit, or remaining capacity is too small, the year is committed. The blender combines mined ore and reclaimed stockpiles to produce plant feed within the allowed grade window.

### Step 9: Export results
After training, the script performs a final greedy rollout and exports results to Excel.

---

## 5. RL architecture

### Mining network
The mining policy uses an **actor-critic** network:
- cluster features are encoded by a neural network
- the actor outputs one logit per cluster
- infeasible actions are masked out
- the critic pools feasible cluster embeddings and combines them with global scalars

### Plant network
The plant policy is a separate neural network that outputs a categorical distribution over plant modes.

### Training objective
- **Mining agent**: Advantage Actor-Critic (A2C) with Generalized Advantage Estimation (GAE)
- **Plant agent**: REINFORCE-style learning on yearly rewards
- joint optimizer over both networks
- gradient clipping for stability

---

## 6. Constraints handled by the environment

The scheduler enforces the following during scheduling:

- cluster precedence
- annual mining capacity
- process capacity
- plant feed grade window
- minimum mining width logic through panel actions
- optional contiguity logic when `Condition=1`
- optional exposed height / maximum allowable depth when enabled
- stockpile capacity checks in the audit

---

## 7. Running the code

### Install dependencies

```bash
pip install numpy pandas scipy torch
```

### Run from command line

```bash
python rl_scheduler_Mar_9_2026.py path/to/config.txt
```

If no config path is supplied, the script uses its built-in default path.

---

## 8. Output files

The main output is an Excel workbook written to:

```txt
output_file_name_path=...
```

### Excel sheets

#### `Results`
Block-by-block scheduling output. Typical columns include:
- `Indices`
- `X`, `Y`, `Z`
- `Grade`
- `Tonnage`
- `Block value` (if present)
- `schedule`

#### `Report`
Yearly production and plant summary. It includes fields such as:
- period
- total tonnage
- ore tonnage
- waste tonnage
- ore grade
- mine-to-process tonnage and grade
- total process tonnage
- process grade
- NPV
- stockpile inflow / reclaim / cumulative stockpile balance
- mining utilisation
- plant utilisation

#### `Audit`
Constraint checking by year, including:
- mining capacity used
- process capacity used
- process grade compliance
- stockpile capacity compliance
- precedence violations
- contiguity indicators
- exposed height checks
- lonely-seed usage flags

#### `Variables`
A record of configuration and reporting metadata.

The script also writes:
- `table_yearly.tex`
- `table_audit.tex`

---

## 9. How to read the results

### Start with `Report`
Use this sheet to understand the yearly production schedule and economics:
- yearly tonnage mined
- ore vs waste mined
- ore grade
- process throughput
- process feed grade
- NPV by year
- stockpile usage and reclaim trends

### Then inspect `Audit`
Use this sheet to verify feasibility:
- no mining capacity overshoot
- no process capacity overshoot
- process grade remains within the defined limits
- no precedence violations
- stockpile capacities are respected
- height constraint is respected when enabled

### Finally inspect `Results`
This is the detailed block schedule. Each block is assigned a mining period in the `schedule` column.

---

## 10. Notes for users

- This code is designed for **cluster-level RL scheduling**, not single-block action selection.
- The config file is the main user interface for changing economics, constraints, capacities, slope angles, and stockpile design.
- If the run becomes infeasible at the start of a year, the code may require increasing mining capacity, splitting clusters more finely, or relaxing geometry/precedence assumptions.
- `Condition=1` activates stricter minimum-width / contiguity behaviour.
- `blender_mode="rl_pure"` is the example mode in the provided config.

---

## 11. Citation / acknowledgement

If you use this repository in academic work, please cite the repository entry that accompanies this scheduler.

