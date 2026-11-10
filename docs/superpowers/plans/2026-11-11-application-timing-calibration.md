# Application-time calibration follow-up

User supplied an agent's screenshot summary: campus i7-14700 runs65.74,
71.41,71.49seconds, reported exit0. No original screenshots, complete logs,
commit/input hashes, or forward traces were supplied in this session. Preserve
those unknowns. Median71.41seconds is a provisional target for this workload,
not a statement of thermal steady state or inference-speed equivalence.

Keep existing phys16-t8 unchanged by default. Add CMM_OMP_THREADS (1..16)
to measure.sh; its output must identify the actual thread count and profile.
Test the actual shell script with small fixture commands, then run phys16-t6
three times serially with the existing cached model,16GiB cap and swap disabled.
Keep all observations, record source/input/dependency provenance and start/end
sensor readings, and compare median/range against the reported campus values.
Do not add downloads, change SDK/application logic, infer a campus commit,
attribute a crash to temperature, or claim a newly adopted equivalence profile.

- [x] Preserve reported campus observations with explicit provenance/unknowns.
- [x] Write failing shell integration tests; implement the thread override.
- [x] Run three isolated6-thread trials and record full outputs and conditions.
- [x] Synchronize docs, run lint/tests and independent review.
- [x] Local commit before final answer; no unsolicited PR operations.

Ruling: if all three6-thread trials remain faster than the provisional target,
also measure5threads three times under the same limits. The user proposed
6threads or nearby; this brackets the target without artificial sleep or quotas.
Do not automatically adopt either candidate as the calibrated SDK profile.
Local temperature readings from measure.sh are the maximum of available hwmon
sensors at start/end, not CPU-only or run peak measurements.

Results:6threads65.12/62.95/63.45s, median63.45;5threads81.80/82.03/81.92s,
median81.92. Both miss provisional71.41s. All six exit0 and save11records,
byte-equal to the checked reference. All samples retained, no profile adopted.
The source/input hashes and existing unrelated name_trie change are recorded.

Verification: make install, make lint (40 files), make test (150 tests), shell
syntax and documentation links. Independent review found no Critical,
Important or Minor findings, reran the4 added harness tests and checked raw
measurements, aggregation, medians, ranges, percentages and source/input hashes.
No model or campus raw-evidence verification was claimed by that reviewer.
