# Hardware comparison probe implementation plan

> Use executing-plans: one implementation batch, tests first, then one review.

**Goal:** Investigate whether the development machine can approximate campus
CPU and memory workloads. This is not an application runtime benchmark.

**User requirements:** No model downloads or weights in the repository. Explain
what is measured. Compare CPU computation, memory throughput and thread scaling
on both machines. Keep existing unrelated edits, commit locally before the final
response, and do not infer new push/PR authorization from local corrections.

**Design:** `tools/hardware_probe.py` has `measure` and `compare` commands.
Use installed NumPy/Pydantic only; no SDK, torch, model or network calls.
Separate serial child processes set BLAS thread variables before importing
NumPy. Profiles are native/default and 1/4/8 threads (within affinity).
Memory read/copy uses two 256 MiB float32 buffers split across Python workers;
float32 matmul uses (1/64/256,1024) x (1024,3072). These are illustrative kernels,
not Qwen inference or a standard STREAM implementation. Record cache, affinity,
software versions, raw integer timings, repeats and variability.

Memory allocation is checked against both host available RAM and cgroup v2
remaining memory. A parent timeout bounds native worker execution. Defaults
are 5 samples of at least 50 ms, total budget 60 s. Array size is adjustable;
small relative-to-cache arrays produce a warning, not a DRAM bandwidth claim.
No hardware settings are changed. Inherited CPU affinity is recorded; a caller
can use taskset to constrain it. Results go to stdout, so only an explicit shell
redirect writes a report. Do not record hostname, username or home paths.

Pydantic validates report shapes; shared JSON parsing rejects duplicate keys
and nonstandard constants. Store elapsed nanoseconds and iteration counts as
integers. Comparison rejects incompatible workload settings, reports software
differences, and compares each candidate against the reference native/default
profile by default. Display ratios and worst deviation; never automatically
claim equivalence or select a new adopted profile.

**Files:** tool, focused tests, this plan, `docs/research/hardware-probe.md`;
update development/campus/EVO-X3 documentation so application timing and
hardware comparison are clearly separated. Retain the correction to offline
campus application commands; do not modify application algorithms or SDK.

- [x] Write failing unit/boundary and real small-worker CLI tests.
- [x] Implement collection, bounded workers and strict comparison.
- [x] Run small smoke, then one default-sized development-machine probe.
- [x] Run make lint and make test; independent final review.
- [x] Synchronize docs, commit authorized changes, keep unrelated name_trie edit.

Sources: NumPy global configuration and show_config documentation; STREAM
reference discusses out-of-cache working sets. Links and limitations go in the
user-facing probe documentation. The default exceeds 4x the summed L3 of the known two-CCD development CPU;
unique shared L3 domains are counted once. Memory throughput counts logical bytes,
not physical bus traffic. Matching these kernels does not prove application
runtime equivalence. Campus results have not been collected for this tool.

Implementation: added14 focused tests; all146 repository tests pass. Cache
accounting now sums unique shared L3 domains rather than one CPU's view.
A preemption regression was observed red (zero iteration batch) and fixed.
Default-sized development run completed; data archived under docs/measurements.
No campus data has been collected and no equivalence is claimed. No model,
SDK, torch, network, dependency or hardware setting change is part of the probe.

Final review: no Critical/Important/Minor findings; reviewer independently ran
all14 probe tests including actual child execution and timeout. make install,
make lint (39 files) and make test (146 tests) passed. No application source or
SDK modifications. The corrected campus application command remains offline
and distinct from the new hardware survey.
