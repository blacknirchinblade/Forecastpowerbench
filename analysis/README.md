# Publication Analysis

`publication.py` loads saved summaries and accuracy tables, standardizes model and
hardware labels, creates publication figures, and implements the documented EAS,
multi-objective alpha, and robust rank-acceptability analyses. It is never imported
by measured inference and therefore cannot affect benchmark timing or telemetry.

The module expects generated result files supplied by the reviewer or experimenter;
no measurements are committed to this anonymous artifact.
