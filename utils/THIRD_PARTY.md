# Third-Party Components

PowerBench does not claim authorship of upstream models, checkpoints, or model
implementations. The following components are either imported from their released
packages/checkpoints or retained under `utils/` because an adapter requires local
runtime source:

| Component | Upstream source | Local role |
|---|---|---|
| UniTS | `mims-harvard/UniTS` | Backbone source and checksum-verified released checkpoint download |
| Kairos | `mldi-lab/Kairos_23m`, `mldi-lab/Kairos_50m` | Forecast backbone source and released checkpoints |
| TinyTimeMixer | `ibm-research/ttm-r3`, `ibm-granite/granite-timeseries-ttm-r2` | Released checkpoints through Granite TSFM |
| Chronos | `amazon/chronos-2`, `amazon/chronos-bolt-base` | Released pipelines and checkpoints |
| Moirai 2.0 | `Salesforce/moirai-2.0-R-small`, Salesforce Uni2TS | Forecast module and released checkpoint |
| TempoPFN | `AutoML-org/TempoPFN` | Runtime source, configuration, and released checkpoint |
| TiRex | `NX-AI/TiRex` | Released package and checkpoint |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | Pinned source and released weights |
| FlowState | `ibm-granite/granite-timeseries-flowstate-r1` | Released checkpoint |
| MOMENT | `AutonLab/MOMENT-1-large` | Released package and checkpoint |
| Sundial | `thuml/sundial-base-128m` | Released package and checkpoint |
| Toto | `Datadog/Toto-Open-Base-1.0`, `Datadog/Toto-2.0-*` | Runtime source and released checkpoints |

The retained UniTS source is covered by `licenses/MIT-UniTS.txt`. Retained Kairos,
TempoPFN, Toto, and Uni2TS runtime source is distributed under Apache-2.0; the
license text is preserved at `licenses/Apache-2.0.txt`, and upstream copyright/SPDX
headers remain in source files where provided.

Reviewers must follow each upstream project's license and model terms. A final public
release should preserve upstream copyright/license notices and add the benchmark's
chosen first-party license before publication.
