# Polymer Series Explorer v8

v8 focuses on seven major improvements to move the tool from pattern screening toward polymer interpretation.

## 1. Better polymer family visualization

The Interactive Series Map still shows every candidate homologous series together, but the detail panel now adds a **Neutral family reconstruction** plot.

- each connected line remains one homologous charge/adduct series
- supported z=1 H, Na, K, and NH4 companion series remain separate observed-ion lines
- linked ion forms are grouped into the same family
- their masses are converted back to neutral mass
- correctly linked adduct forms should collapse onto the same neutral oligomer distribution

## 2. End-group / intercept analysis

For every selected series, the software fits:

`mass = intercept + n × repeat mass`

It reports:

- fitted repeat mass
- repeat-mass ppm error
- fit R²
- observed index-zero intercept
- residual mass modulo the repeat

The residual modulo the repeat is an end-group clue. It is intentionally not presented as an absolute end-group identity because the observed first member is not necessarily DP 0.

## 3. Candidate formulas for unknown repeat masses

Discovery mode can now infer candidate elemental formulas for an unknown repeat mass.

The user can choose allowed elements, with a default set of:

- C
- H
- N
- O
- Si

Candidate formulas are ranked by mass error and chemical plausibility and are clearly labeled as **candidate formulas**, not identifications.

## 4. Stronger discovery scoring

`Find any homologous series` now scores each unknown repeat candidate using:

- direct recurring mass-difference support
- number of connected candidate series
- number of assigned features
- consecutive homolog continuity
- mass accuracy
- harmonic / subharmonic behavior
- whether the shift resembles a selected adduct-to-adduct mass difference
- whether it duplicates a targeted repeat unit

Each discovered repeat receives a 0–100 discovery score.

## 5. Series confidence score

Each homologous series receives a 0–100 confidence score based on:

- number of members
- consecutive homolog continuity
- mass accuracy
- retention-time trend
- companion adduct support
- targeted versus discovery repeat support

Confidence is shown as High, Medium, or Low and the component scores are available in the series detail panel.

## 6. Automatic charge detection

Automatic charge detection is enabled by default.

The software tests charge states up to a user-selected maximum and then removes duplicate charge hypotheses when nearly the same features are explained by several charges. The preferred assignment emphasizes consecutive repeat indices and mass accuracy.

This allows, for example, a 44.0262 Da PEG repeat to be recognized from approximately 22.0131 m/z spacing for a z=2 series.

## 7. Neutral mass reconstruction and adduct collapsing

Supported z=1 adduct-linked series are reconstructed to neutral mass:

- [M+H]+
- [M+Na]+
- [M+K]+
- [M+NH4]+
- [M-H]-

The observed ion series stay separate so the evidence remains inspectable, while the underlying neutral family can be viewed together.

## Important current limitation

The common adduct library currently describes singly charged adduct forms. Automatic charge-state detection works for z>1 homologous spacing, but z>1 adduct deconvolution is not forced into the z=1 adduct model.

## Existing capabilities retained

- CH2 / methylene repeat search at 14.01565 Da
- multiple targeted repeat units in one run
- custom repeat formula
- discovery of unknown homologous series
- interactive all-series plot
- click-to-select series
- linked adduct evidence
- rerun without reloading the data
- CSV exports for selected series, linked family, all assignments, summaries, adduct relationships, and discovered repeat masses

## Windows launch

Double click:

`launch_windows.bat`

or run:

```cmd
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```
