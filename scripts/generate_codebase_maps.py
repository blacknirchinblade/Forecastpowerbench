#!/usr/bin/env python3
"""Generate complete, reviewable codebase maps from PowerBench source files."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "codebase_maps"
FIRST_PARTY_DIRS = ("benchmark", "experiments", "models", "scripts", "tests")
RUNTIME_DIRS = ("benchmark", "experiments", "models", "scripts")


@dataclass
class Parameter:
    """Define the Parameter type and its associated runtime state or contract.
    
    Attributes:
        annotation: Annotation state retained by each instance.
        default: Default state retained by each instance.
        description: Description state retained by each instance.
        kind: Kind state retained by each instance.
        name: Name state retained by each instance.
    """
    name: str
    kind: str
    annotation: str
    default: str
    description: str


@dataclass
class Symbol:
    """Define the Symbol type and its associated runtime state or contract.
    
    Attributes:
        attributes: Attributes state retained by each instance.
        bases: Bases state retained by each instance.
        calls: Calls state retained by each instance.
        kind: Kind state retained by each instance.
        line: Line state retained by each instance.
        parameters: Parameters state retained by each instance.
        qualified_name: Qualified name state retained by each instance.
        returns: Returns state retained by each instance.
        signature: Signature state retained by each instance.
        summary: Summary state retained by each instance.
    """
    qualified_name: str
    kind: str
    line: int
    signature: str
    summary: str
    parameters: list[Parameter] = field(default_factory=list)
    returns: str = "Not applicable."
    bases: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    calls: set[str] = field(default_factory=set)


@dataclass
class Module:
    """Define the Module type and its associated runtime state or contract.
    
    Attributes:
        calls: Calls state retained by each instance.
        imports: Imports state retained by each instance.
        name: Name state retained by each instance.
        path: Filesystem path consumed by the operation.
        summary: Summary state retained by each instance.
        symbols: Symbols state retained by each instance.
    """
    path: Path
    name: str
    summary: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    calls: set[tuple[str, str]] = field(default_factory=set)


def clean_text(value: str | None, fallback: str = "No module docstring; see the source.") -> str:
    """Collapse whitespace and escape table delimiters in generated documentation text.
    
    Args:
        value (str | None): Configuration or runtime value named value.
        fallback (str): Configuration or runtime value named fallback. Defaults to 'No module
                        docstring; see the source.'.
    
    Returns:
        str: Result produced by this operation.
    """
    if not value:
        return fallback
    return " ".join(value.strip().split()).replace("|", "\\|")


def doc_summary(value: str | None) -> str:
    """Extract the explanatory paragraph before structured docstring sections.
    
    Args:
        value (str | None): Full source docstring.
    
    Returns:
        str: Compact purpose text for indexes and mind maps.
    """
    if not value:
        return ""
    lines = []
    headings = {
        "Args:", "Arguments:", "Parameters:", "Returns:", "Yields:",
        "Attributes:", "Bases:", "Raises:", "Notes:",
    }
    for raw in value.strip().splitlines():
        line = raw.strip()
        if line in headings or (not line and lines):
            break
        if line:
            lines.append(line)
    return clean_text(" ".join(lines), "")


def safe_unparse(node: ast.AST | None, fallback: str = "unspecified") -> str:
    """Convert an AST node back to source text with a deterministic fallback.
    
    Args:
        node (ast.AST | None): Configuration or runtime value named node.
        fallback (str): Configuration or runtime value named fallback. Defaults to 'unspecified'.
    
    Returns:
        str: Result produced by this operation.
    """
    if node is None:
        return fallback
    try:
        return ast.unparse(node).replace("|", "\\|")
    except Exception:
        return fallback


PARAMETER_PURPOSES = {
    "self": "Instance receiving the method call.",
    "cls": "Class receiving the class-method call.",
    "batch_size": "Number of source-local windows passed to one model call.",
    "config": "Validated benchmark protocol and output configuration.",
    "context_len": "Number of historical points supplied for each forecast window.",
    "ctx": "Context length for the current configuration.",
    "data_dir": "Root directory searched recursively for source CSV files.",
    "data_path": "Dataset directory passed to the runner.",
    "dataset": "Window dataset used to construct deterministic batches.",
    "dataset_name": "Stable dataset identifier written to output metadata.",
    "device": "Requested execution device or device selector.",
    "device_profile": "Hardware profile responsible for setup, verification, and restoration.",
    "duration_sec": "Requested duration in seconds.",
    "end_ns": "Inclusive measurement end timestamp in nanoseconds.",
    "hor": "Forecast horizon for the current configuration.",
    "horizon_len": "Number of future points requested from the model.",
    "inputs": "Batched model context with shape `(batch, context)`.",
    "interval_ms": "Telemetry sampling interval in milliseconds.",
    "model_name": "Registered model adapter name.",
    "num_workers": "PyTorch DataLoader worker-process count.",
    "output_dir": "Directory containing run artifacts.",
    "path": "Filesystem path consumed by the operation.",
    "pattern": "Recursive source-file glob pattern.",
    "power_monitor": "Telemetry backend sampled around measured repeats.",
    "power_mode": "Requested device power-mode identifier.",
    "rows": "Per-repeat or summary records to process.",
    "seed": "Root random seed used for deterministic execution.",
    "sources": "Validated independent source series; sources are never concatenated before windowing.",
    "start_ns": "Inclusive measurement start timestamp in nanoseconds.",
    "target_column": "Optional explicit forecast-target column name.",
    "time_column": "Optional explicit timestamp column name.",
    "values": "Values to normalize, summarize, or serialize.",
    "indices": "Integer positions requested from the deterministic window or source array.",
    "columns": "Available dataset column labels considered during resolution.",
    "requested": "Explicit column name requested by the operator, when provided.",
    "candidates": "Ordered fallback column names accepted by the resolver.",
    "excluded_names": "Source filenames excluded from recursive discovery.",
    "max_sources": "Optional maximum number of source files retained after stable sorting.",
    "stride": "Number of source points between consecutive window starts.",
    "max_windows_per_source": "Optional deterministic cap applied independently to each source.",
    "include_targets": "Whether each dataset item includes its future target window.",
    "max_pool_windows": "Maximum unique windows materialized in the prepared benchmark pool.",
    "cache_dir": "Optional directory for validated preprocessed source caches.",
    "capture_output": "Whether subprocess standard output and error are captured.",
    "category": "Hardware category represented by this device profile.",
    "output_final_state": "Whether the recurrent kernel also returns its final hidden state.",
    "elementwise_affine": "Whether normalization owns learnable scale and bias parameters.",
    "residual_in_fp32": "Whether residual accumulation is promoted to float32 precision.",
    "model_path": "Model repository or local artifact identifier loaded by the adapter.",
    "checkpoint_path": "Path to the pretrained checkpoint loaded by the adapter.",
    "freq": "Frequency label forwarded to model-specific time-series metadata.",
    "prediction_batch_size": "Fixed batch size used only for deterministic Phase-1 prediction generation.",
    "dataset_sha256": "Optional dataset fingerprint required for resume compatibility.",
    "policy": "Named validation or preprocessing policy recorded in metadata.",
    "exc": "Exception whose message and type are inspected.",
    "record": "Logging record being filtered or emitted.",
    "batch_index": "Zero-based index of the prepared benchmark batch.",
    "worker_id": "Zero-based DataLoader worker identifier assigned by PyTorch.",
    "clocks_locked": "Whether the device clocks were successfully locked.",
    "mode_id": "Device-specific power-mode identifier to apply or verify.",
    "command": "Command and arguments executed without a shell.",
    "cache_key": "Content-derived identity used to validate cached source data.",
    "cache_path": "Path of the prepared source cache artifact.",
    "source_path": "Path of the independent source CSV being processed.",
    "source": "One independently validated source series.",
    "predictions": "Raw adapter output to normalize or persist.",
    "expected_repeats": "Number of measured repeats required by the protocol.",
    "baseline_bytes": "Swap usage captured before the measured operation.",
    "secs": "Duration in seconds.",
    "desc": "Short user-facing label for progress output.",
}


PURPOSE_OVERRIDES = {
    "benchmark.config.BenchmarkConfig": "Validate and carry the complete two-phase benchmark protocol.",
    "benchmark.config.ResultRow": "Represent one measured repeat or terminal configuration using the versioned result schema.",
    "benchmark.data.SourceSeries": "Hold one independently validated and median-imputed source series with timestamps, observed mask, and provenance.",
    "benchmark.data.MultiSourceWindowDataset": "Map deterministic global indices to complete source-local context and target windows without cross-source leakage.",
    "benchmark.data.PreparedBenchmarkStream": "Materialize a bounded contiguous input pool before timing and expose deterministic full-batch views.",
    "benchmark.runner.ProcessMemorySampler": "Sample process-tree resident memory in parallel and return the peak RSS for a measured repeat.",
    "benchmark.runner.BenchmarkRunner": "Coordinate data preparation, prediction generation, device control, timed inference, telemetry, aggregation, and persistence.",
    "benchmark.runner.BenchmarkRunner._run_prediction_pass": "Generate untimed deterministic forecasts for one model/context/horizon and save one NPZ per source.",
    "benchmark.runner.BenchmarkRunner._run_source_directory": "Execute the selected prediction or benchmark phase over the configured model and hardware loops.",
    "benchmark.runner.BenchmarkRunner._run_source_batch_size": "Run one model/context/horizon/batch-size configuration, including preload, baseline, warmups, measured repeats, telemetry, and summary persistence.",
    "benchmark.monitors.base.PowerMonitor": "Define the start/stop telemetry contract implemented by each hardware backend.",
    "models.base.BaseModel": "Define the common batched point-forecast adapter contract.",
    "models.factory.get_model": "Lazily import and instantiate the registered device-appropriate model adapter.",
    "benchmark.runner._is_batch_limit_runtime": "Classify runtime failures that indicate an unsupported or memory-exhausting effective batch size.",
    "benchmark.runner.configure_torch_cpu_threads": "Apply the benchmark's PyTorch intra-op and inter-op CPU thread settings once at startup.",
    "benchmark.runner.ProcessMemorySampler._loop": "Sample parent-and-worker RSS until collection is stopped, retaining the highest observation.",
    "benchmark.runner.progress": "Construct a terminal-safe tqdm progress bar using the benchmark's common refresh and display policy.",
    "benchmark.runner.clear_gpu_memory": "Release Python and CUDA caches between isolated configurations without entering a measured region.",
    "benchmark.data.SourceSeries.observed_at": "Return whether requested source positions were originally observed rather than imputed.",
    "benchmark.data.SourceSeries.timestamps_at": "Return timestamps at requested positions while preserving source-local alignment.",
    "benchmark.data.MultiSourceWindowDataset.__getitem__": "Build one complete source-local context/target window and its provenance metadata.",
    "benchmark.data.MultiSourceWindowDataset.__getitems__": "Build several indexed windows as one vectorized NumPy batch without crossing source boundaries.",
    "benchmark.data.PreparedBenchmarkStream.batch": "Return one contiguous full batch from the prebuilt pool, wrapping deterministically when repetition is required.",
    "benchmark.monitors.stats.summarize_power_interval": "Clip timestamped telemetry to the measured interval and derive power, utilization, temperature, memory, and trapezoidal-energy statistics.",
    "experiments.forecasting.configure_logging": "Route concise milestone messages to the terminal and complete diagnostic output to the model log.",
    "experiments.forecasting.auto_detect_device": "Identify the active benchmark hardware category from platform and accelerator capabilities.",
    "experiments.forecasting.main": "Parse the command line, establish deterministic runtime policy, and execute the configured benchmark runner.",
    "benchmark.runner.BenchmarkRunner._initialize_dict_csv": "Create a schema-only CSV atomically when the requested result file does not yet exist.",
    "benchmark.runner.BenchmarkRunner._append_dict_rows": "Append schema-aligned records to a CSV with one flush after a completed configuration.",
    "benchmark.runner.BenchmarkRunner._cpu_ram_limit_gb": "Return the configured safe process-tree RAM ceiling for CPU execution.",
    "benchmark.runner.BenchmarkRunner._gpu_memory_limit_gb": "Return the safe accelerator-memory ceiling after applying configured headroom.",
    "benchmark.runner.BenchmarkRunner._remember_cpu_peak": "Retain a successful CPU peak-RSS observation for the next-batch memory projection.",
    "benchmark.runner.BenchmarkRunner._remember_gpu_peak": "Retain a successful CUDA peak-reserved observation for the next-batch memory projection.",
    "benchmark.runner.BenchmarkRunner._shutdown_dataloader_workers": "Shut down persistent DataLoader workers explicitly so model processes release resources promptly.",
    "benchmark.runner.BenchmarkRunner._execution_device": "Return the concrete adapter device string selected for the current hardware category.",
    "benchmark.runner.BenchmarkRunner._safe_path_part": "Sanitize arbitrary metadata into one stable filesystem path component.",
    "benchmark.runner.BenchmarkRunner._prediction_npz_path": "Construct the model/source/context/horizon path for one Phase-1 prediction artifact.",
    "benchmark.runner.BenchmarkRunner._run_source_batch_size.raise_config_timeout": "Raise the configuration timeout from the process alarm handler.",
    "benchmark.data._segment_bounds": "Return contiguous index ranges separated by timestamp-cadence discontinuities.",
    "benchmark.data._source_cache_key": "Hash source identity and preprocessing policy into a cache-validation key.",
    "benchmark.data._cache_path": "Return the deterministic NPZ cache path for one source key.",
    "benchmark.data.MultiSourceWindowDataset._empty_batch": "Return correctly typed empty tensors and metadata for a zero-length vectorized request.",
    "benchmark.data.PreparedBenchmarkStream.repeated": "Return whether the fixed benchmark demand exceeds the unique preloaded window pool.",
    "benchmark.monitors.base.PowerMonitor.get_raw_line": "Return the most recent backend telemetry line for concise diagnostics.",
    "benchmark.monitors.nvml.NVMLPowerMonitor.get_raw_line": "Return a concise representation of the most recent NVML telemetry sample.",
    "benchmark.monitors.jetson.JetsonPowerMonitor.get_raw_line": "Return the most recent raw tegrastats line for diagnostics.",
    "benchmark.monitors.rapl.RAPLPowerMonitor.get_raw_line": "Return a concise representation of the most recent RAPL sample.",
    "benchmark.monitors.sysfs.SysfsPowerMonitor.get_raw_line": "Return a concise representation of the most recent sysfs sample.",
    "benchmark.monitors.null.NullPowerMonitor.get_raw_line": "Return an empty diagnostic line because this backend collects no telemetry.",
    "benchmark.monitors.stats._trapezoid": "Integrate timestamped samples using NumPy's available trapezoidal-rule implementation.",
    "benchmark.monitors.rapl.RAPLPowerMonitor._readable_rapl_paths": "Return RAPL energy-counter paths that exist and are readable by the benchmark process.",
    "benchmark.monitors.sysfs.SysfsPowerMonitor._rpi_throttle_event": "Return whether Raspberry Pi throttle flags were active or changed during the interval.",
    "models.chronos2._install_identity_collate.inference_data_loader": "Create the Chronos inference DataLoader while applying identity collation only to prebatched tensors.",
    "models.tempopfn_cpu.dummy_jit": "Provide an import-compatible no-op replacement for Triton's JIT decorator on CPU.",
    "models.tempopfn_cpu.dummy_jit.decorate": "Attach Triton-style argument metadata and return the original CPU callable unchanged.",
    "models.tempopfn_cpu.dummy_triton_language_attribute": "Provide inert Triton-language placeholders needed while importing the CPU adapter.",
    "models.tempopfn_cpu.dummy_libdevice_attribute": "Provide inert libdevice callables needed while importing the CPU adapter.",
    "models.tempopfn_cpu._cpu_find_spec": "Hide CUDA-only module specifications while resolving TempoPFN dependencies on CPU.",
    "scripts.benchmark_resume._model_log_completed": "Return whether a model log contains the explicit Phase-2 completion marker.",
    "scripts.generate_codebase_maps.clean_text": "Collapse whitespace and escape table delimiters in generated documentation text.",
    "scripts.generate_codebase_maps.safe_unparse": "Convert an AST node back to source text with a deterministic fallback.",
    "scripts.generate_codebase_maps.words_from_identifier": "Convert snake_case or CamelCase identifiers into readable lowercase words.",
    "scripts.generate_codebase_maps.infer_returns": "Describe a callable's return contract from its annotation and implementation shape.",
    "scripts.generate_codebase_maps.class_metadata": "Extract class bases and statically visible instance or class fields.",
    "scripts.generate_codebase_maps.module_name": "Convert a repository-relative Python path into its importable module name.",
    "scripts.generate_codebase_maps.import_target": "Resolve an absolute module target from an AST relative-import declaration.",
    "scripts.generate_codebase_maps.scan_module": "Parse one Python module into documented symbols, imports, fields, and static call edges.",
    "scripts.generate_codebase_maps.first_party_files": "Return every maintained Python file included in documentation coverage.",
    "scripts.generate_codebase_maps.shell_symbols": "Extract shell function names and source lines from one launcher script.",
    "scripts.generate_codebase_maps.role_for": "Classify a retained source path by its repository responsibility.",
    "scripts.generate_codebase_maps.parameter_compact": "Format one parameter's name, type, and default for a compact mind-map node.",
    "scripts.generate_codebase_maps.subsystem": "Return the architecture subsystem assigned to a first-party module.",
    "scripts.prediction_resume._prediction_files": "Return all prediction artifacts currently stored for one model.",
    "scripts.prediction_resume._relative_prediction_files": "Return model prediction artifact paths relative to the prediction root.",
    "scripts.supplemental_batches._matches_requested_config": "Return whether a result row belongs to the requested supplemental sweep.",
    "scripts.validate_results._finite_positive": "Return whether a value parses to a finite scalar greater than zero.",
    "tests.test_benchmark_resume.BenchmarkResumeTests._row": "Construct one synthetic Phase-2 result row for resume tests.",
    "tests.test_benchmark_resume.BenchmarkResumeTests._complete_log": "Write a synthetic model log containing the benchmark completion marker.",
    "tests.test_chronos2_adapter.Chronos2AdapterTests.test_identity_collate_only_applies_to_prebatched_chronos_loader.fake_loader": "Capture Chronos DataLoader arguments and return a lightweight fake loader.",
    "tests.test_gpu_memory_guard.GPUMemoryGuardTests._runner": "Construct a minimal runner with controlled GPU-memory observations.",
    "tests.test_gpu_preflight.GPUPreflightTests._runner": "Construct a runner with mocked GPU-utilization samples for preflight tests.",
    "tests.test_prepared_benchmark.PreparedBenchmarkStreamTests._dataset": "Construct a deterministic synthetic window dataset for prepared-stream tests.",
    "tests.test_supplemental_batches.SupplementalBatchTests._row": "Construct one synthetic result row for supplemental-sweep tests.",
    "tests.test_ttm_r2_adapter.TTMR2AdapterTests._load_adapter.fake_get_model": "Return the fake TTM checkpoint object used to isolate adapter behavior.",
}


def words_from_identifier(value: str) -> str:
    """Convert snake_case or CamelCase identifiers into readable lowercase words.
    
    Args:
        value (str): Configuration or runtime value named value.
    
    Returns:
        str: Result produced by this operation.
    """
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return value.strip("_").replace("_", " ").lower()


def infer_symbol_purpose(qualified: str, short_name: str, kind: str, doc: str | None) -> str:
    """Resolve concise purpose text from source docs or conservative naming rules.
    
    Args:
        qualified (str): Fully qualified symbol name.
        short_name (str): Module-local or class-local symbol name.
        kind (str): Symbol category: class, function, or method.
        doc (str | None): Full source docstring when declared.
    
    Returns:
        str: One compact purpose paragraph.
    """
    if doc:
        return doc_summary(doc)
    if qualified in PURPOSE_OVERRIDES:
        return PURPOSE_OVERRIDES[qualified]
    name = short_name.rsplit(".", 1)[-1]
    subject = words_from_identifier(name)
    owner = short_name.rsplit(".", 1)[0] if "." in short_name else "the module"
    if kind == "class":
        return f"Define the `{short_name}` type and its associated runtime state or contract."
    if name == "__init__":
        return f"Initialize `{owner}` and capture its validated configuration or runtime state."
    if name == "__len__":
        return f"Return the number of addressable items exposed by `{owner}`."
    if name in {"__call__", "forward"}:
        return f"Execute the callable inference behavior implemented by `{owner}`."
    if name == "__getitem__":
        return f"Return the indexed item exposed by `{owner}`."
    if name == "__getitems__":
        return f"Return multiple indexed items exposed by `{owner}` using its vectorized path."
    if name in {"setUp", "setup_method"}:
        return f"Create isolated fixtures required by `{owner}` tests."
    if name in {"tearDown", "teardown_method"}:
        return f"Release fixtures and temporary artifacts created by `{owner}` tests."
    if name.startswith("test_"):
        return f"Regression-check that {words_from_identifier(name[5:])}."
    actions = (
        ("_run_", "Execute"), ("run", "Execute"), ("_load_", "Load"),
        ("load_", "Load"), ("_save_", "Persist"), ("save_", "Persist"),
        ("_write_", "Write"), ("write_", "Write"), ("_parse_", "Parse"),
        ("parse_", "Parse"), ("_format_", "Format"), ("format_", "Format"),
        ("_normalize_", "Normalize"), ("normalize_", "Normalize"),
        ("_validate_", "Validate"), ("validate_", "Validate"),
        ("_check_", "Check"), ("check_", "Check"),
        ("_should_", "Determine whether to"), ("should_", "Determine whether to"),
        ("_record_", "Record"), ("record_", "Record"),
        ("_make_", "Construct"), ("make_", "Construct"),
        ("_read_", "Read"), ("read_", "Read"),
        ("_resolve_", "Resolve"), ("resolve_", "Resolve"),
        ("_align_", "Align"), ("align_", "Align"),
        ("_summarize_", "Summarize"), ("summarize_", "Summarize"),
        ("_configure_", "Configure"), ("configure_", "Configure"),
        ("_discover_", "Discover"), ("discover_", "Discover"),
        ("_find_", "Find"), ("find_", "Find"),
        ("_query_", "Query"), ("query_", "Query"),
        ("_capture", "Capture"), ("capture", "Capture"),
        ("_calculate_", "Calculate"), ("calculate_", "Calculate"),
        ("_estimate_", "Estimate"), ("estimate_", "Estimate"),
        ("_count_", "Count"), ("count_", "Count"),
        ("_prepare_", "Prepare"), ("prepare_", "Prepare"),
        ("_move_", "Move"), ("move_", "Move"),
        ("_filter_", "Filter"), ("filter_", "Filter"),
        ("_split_", "Split"), ("split_", "Split"),
        ("_atomic_", "Atomically update"), ("atomic_", "Atomically update"),
        ("_measure_", "Measure"), ("measure_", "Measure"),
        ("_apply_", "Apply"), ("apply_", "Apply"),
        ("_verify_", "Verify"), ("verify_", "Verify"),
        ("_initialize_", "Initialize"), ("initialize_", "Initialize"),
        ("_append_", "Append"), ("append_", "Append"),
        ("_shutdown_", "Shut down"), ("shutdown_", "Shut down"),
        ("_install_", "Install"), ("install_", "Install"),
        ("_extract_", "Extract"), ("extract_", "Extract"),
        ("_convert_", "Convert"), ("convert_", "Convert"),
        ("_build_", "Build"), ("build_", "Build"),
        ("_compute_", "Compute"), ("compute_", "Compute"),
        ("_collect_", "Collect"), ("collect_", "Collect"),
        ("_create_", "Create"), ("create_", "Create"),
        ("_select_", "Select"), ("select_", "Select"),
        ("_remove_", "Remove"), ("remove_", "Remove"),
        ("_merge_", "Merge"), ("merge_", "Merge"),
        ("_ensure_", "Ensure"), ("ensure_", "Ensure"),
        ("_log_", "Log"), ("log_", "Log"),
    )
    if name == "forecast":
        return f"Run `{owner}` point forecasting for one batched context input."
    if name == "start":
        return f"Start `{owner}` collection or lifecycle state."
    if name == "stop":
        return f"Stop `{owner}` collection or lifecycle state and return its final result when applicable."
    if name == "setup":
        return f"Prepare `{owner}` hardware or runtime state before benchmark execution."
    if name == "teardown":
        return f"Restore host state changed by `{owner}` after benchmark execution."
    if name in {"main", "cli"}:
        return f"Parse command-line arguments and execute the `{owner}` workflow."
    if name == "__enter__":
        return f"Enter the compatibility context represented by `{owner}`."
    if name == "__exit__":
        return f"Exit the compatibility context represented by `{owner}` without suppressing failures."
    if name in {"to", "eval"}:
        return f"Mirror the framework `{name}` lifecycle method for the `{owner}` test or compatibility object."
    if name in {"predict", "predict_quantiles"}:
        return f"Produce deterministic fake or adapter predictions through `{owner}`."
    if name == "from_pretrained":
        return f"Construct `{owner}` through its pretrained-model compatibility entry point."
    if name in {"emit", "filter"}:
        verb = "Emit" if name == "emit" else "Select"
        return f"{verb} a logging record according to `{owner}` console policy."
    if name.endswith("_loop") or name == "loop":
        return f"Run the background collection loop owned by `{owner}` until it is stopped."
    if name.startswith(("is_", "_is_", "has_", "_has_", "supports_", "_supports_")) or name.endswith(("_complete", "_present", "_matches", "_finite")):
        return f"Return whether {subject.lstrip('_')} for `{owner}`."
    for prefix, verb in actions:
        if name.startswith(prefix):
            remainder = words_from_identifier(name[len(prefix):]) or subject
            return f"{verb} {remainder} for `{owner}`."
    return f"Compute or provide `{subject}` as part of `{owner}`."


def parse_doc_parameters(doc: str | None) -> dict[str, str]:
    """Parse Google- and Sphinx-style parameters, including wrapped descriptions.
    
    Args:
        doc (str | None): Configuration or runtime value named doc.
    
    Returns:
        dict[str, str]: Named values produced by this operation.
    """
    descriptions: dict[str, str] = {}
    if not doc:
        return descriptions
    in_args = False
    current_name: str | None = None
    for raw in doc.splitlines():
        line = raw.strip()
        sphinx = re.match(r":param\s+([*\w]+)\s*:\s*(.+)", line)
        if sphinx:
            current_name = sphinx.group(1).lstrip("*")
            descriptions[current_name] = clean_text(sphinx.group(2))
            continue
        if line in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            current_name = None
            continue
        if in_args:
            match = re.match(r"([*\w]+)(?:\s*\([^)]*\))?\s*:\s*(.+)", line)
            if match:
                current_name = match.group(1).lstrip("*")
                descriptions[current_name] = clean_text(match.group(2))
            elif line and raw.startswith((" ", "\t")) and current_name:
                descriptions[current_name] = clean_text(
                    f"{descriptions[current_name]} {line}"
                )
            elif line and not raw.startswith((" ", "\t")):
                in_args = False
                current_name = None
    return descriptions


def extract_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Parameter]:
    """Extract parameters for the module.
    
    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Configuration or runtime value named node.
    
    Returns:
        list[Parameter]: Result produced by this operation.
    """
    descriptions = parse_doc_parameters(ast.get_docstring(node))
    result: list[Parameter] = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    posonly_count = len(node.args.posonlyargs)
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        kind = "positional-only" if index < posonly_count else "positional-or-keyword"
        result.append(
            Parameter(
                name=argument.arg,
                kind=kind,
                annotation=safe_unparse(argument.annotation, "unannotated"),
                default=safe_unparse(default, "required"),
                description=descriptions.get(
                    argument.arg,
                    PARAMETER_PURPOSES.get(
                        argument.arg,
                        f"Configuration or runtime value named `{argument.arg}`.",
                    ),
                ),
            )
        )
    if node.args.vararg:
        argument = node.args.vararg
        result.append(Parameter(argument.arg, "variadic positional", safe_unparse(argument.annotation, "unannotated"), "empty", descriptions.get(argument.arg, "Additional positional arguments forwarded by the callable.")))
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        result.append(
            Parameter(
                argument.arg,
                "keyword-only",
                safe_unparse(argument.annotation, "unannotated"),
                safe_unparse(default, "required"),
                descriptions.get(argument.arg, PARAMETER_PURPOSES.get(argument.arg, f"Keyword-only configuration named `{argument.arg}`.")),
            )
        )
    if node.args.kwarg:
        argument = node.args.kwarg
        result.append(Parameter(argument.arg, "variadic keyword", safe_unparse(argument.annotation, "unannotated"), "empty", descriptions.get(argument.arg, "Additional keyword options forwarded to the underlying implementation.")))
    return result


def infer_returns(node: ast.FunctionDef | ast.AsyncFunctionDef, short_name: str) -> str:
    """Describe a callable's return contract from its annotation and implementation shape.
    
    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Configuration or runtime value named node.
        short_name (str): Configuration or runtime value named short_name.
    
    Returns:
        str: Result produced by this operation.
    """
    annotation = safe_unparse(node.returns, "")
    if annotation:
        normalized = annotation.replace("\\|", "|")
        if normalized in {"None", "NoneType"}:
            return "None; the operation updates state, validates input, or persists output in place."
        if normalized == "bool":
            return "bool: Whether the condition described by this callable holds."
        if normalized in {"float", "int"}:
            return f"{normalized}: Computed scalar described by this callable."
        if "ndarray" in normalized:
            return f"{normalized}: NumPy data produced by this operation."
        if normalized.startswith(("Dict", "dict")):
            return f"{normalized}: Named values produced by this operation."
        return f"{normalized}: Result produced by this operation."
    name = short_name.rsplit(".", 1)[-1]
    if name == "__init__":
        return "`None`; initializes the instance in place."
    if name == "forecast":
        return "A pair containing normalized point forecasts and adapter-reported model-call duration."
    has_value = any(
        isinstance(item, ast.Return) and item.value is not None
        for item in ast.walk(node)
    )
    return "Runtime value with no declared return annotation; see the implementation and callers." if has_value else "`None`; the callable operates through state changes, validation, or persistence."


def class_metadata(node: ast.ClassDef) -> tuple[list[str], list[str]]:
    """Extract class bases and statically visible instance or class fields.
    
    Args:
        node (ast.ClassDef): Configuration or runtime value named node.
    
    Returns:
        tuple[list[str], list[str]]: Result produced by this operation.
    """
    bases = [safe_unparse(base) for base in node.bases]
    attributes: set[str] = set()
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            text = f"{child.target.id}: {safe_unparse(child.annotation)}"
            if child.value is not None:
                text += f" = {safe_unparse(child.value)}"
            attributes.add(text)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
            for item in ast.walk(child):
                if isinstance(item, (ast.Assign, ast.AnnAssign)):
                    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                    for target in targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                            attributes.add(target.attr)
    return bases, sorted(attributes)


def module_name(path: Path) -> str:
    """Convert a repository-relative Python path into its importable module name.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
    
    Returns:
        str: Result produced by this operation.
    """
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format signature for the module.
    
    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Configuration or runtime value named node.
    
    Returns:
        str: Result produced by this operation.
    """
    arguments = []
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for argument, default in zip(positional, defaults):
        text = argument.arg
        if default is not None:
            try:
                text += f"={ast.unparse(default)}"
            except Exception:
                text += "=..."
        arguments.append(text)
    if node.args.vararg:
        arguments.append(f"*{node.args.vararg.arg}")
    arguments.extend(argument.arg for argument in node.args.kwonlyargs)
    if node.args.kwarg:
        arguments.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(arguments)})"


def import_target(current: str, node: ast.ImportFrom) -> str:
    """Resolve an absolute module target from an AST relative-import declaration.
    
    Args:
        current (str): Configuration or runtime value named current.
        node (ast.ImportFrom): Configuration or runtime value named node.
    
    Returns:
        str: Result produced by this operation.
    """
    if node.level == 0:
        return node.module or ""
    package = current.split(".")[:-1]
    keep = max(0, len(package) - node.level + 1)
    prefix = package[:keep]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def definition_nodes(tree: ast.AST) -> list[tuple[ast.AST, str, str]]:
    """Collect every top-level or nested class/function with a lexical name.
    
    Args:
        tree (ast.AST): Parsed module syntax tree.
    
    Returns:
        list[tuple[ast.AST, str, str]]: Nodes paired with qualified local names and symbol kinds.
    """
    symbols: list[tuple[ast.AST, str, str]] = []

    class Collector(ast.NodeVisitor):
        """Track lexical owners while traversing nested declarations.
        
        Bases:
            ast.NodeVisitor
        
        Attributes:
            stack: Stack state retained by each instance.
        """

        def __init__(self) -> None:
            """Initialize definition_nodes.Collector and capture its validated configuration or runtime state.
            
            Returns:
                None; initializes the visitor in place.
            """
            self.stack: list[tuple[str, str]] = []

        def _record(self, node: ast.AST, node_kind: str) -> None:
            """Record one definition and recursively inspect its body.
            
            Args:
                node (ast.AST): Class or function declaration being recorded.
                node_kind (str): Either class or function.
            
            Returns:
                None; appends to the enclosing ``symbols`` collection.
            """
            local_name = ".".join([*(name for name, _ in self.stack), node.name])
            kind = (
                "class"
                if node_kind == "class"
                else "method" if self.stack and self.stack[-1][1] == "class" else "function"
            )
            symbols.append((node, local_name, kind))
            self.stack.append((node.name, node_kind))
            self.generic_visit(node)
            self.stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Record a class declaration and its nested definitions.
            
            Args:
                node (ast.ClassDef): Class declaration being visited.
            
            Returns:
                None; updates the collected symbol inventory.
            """
            self._record(node, "class")

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Record a synchronous function and its nested definitions.
            
            Args:
                node (ast.FunctionDef): Function declaration being visited.
            
            Returns:
                None; updates the collected symbol inventory.
            """
            self._record(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Record an asynchronous function and its nested definitions.
            
            Args:
                node (ast.AsyncFunctionDef): Function declaration being visited.
            
            Returns:
                None; updates the collected symbol inventory.
            """
            self._record(node, "function")

    Collector().visit(tree)
    return symbols


def scan_module(path: Path) -> Module:
    """Parse one Python module into documented symbols, imports, fields, and static call edges.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
    
    Returns:
        Module: Result produced by this operation.
    """
    text = path.read_text(errors="replace")
    tree = ast.parse(text, filename=str(path))
    name = module_name(path)
    module = Module(path=path, name=name, summary=clean_text(ast.get_docstring(tree)))
    aliases: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module.imports.add(alias.name)
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            target = import_target(name, node)
            if target:
                module.imports.add(target)
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{target}.{alias.name}".strip(".")

    symbol_nodes = definition_nodes(tree)

    local_names = {short.split(".")[-1] for _, short, _ in symbol_nodes}
    for node, short_name, kind in symbol_nodes:
        signature = short_name
        parameters: list[Parameter] = []
        returns = "Not applicable to a class declaration."
        bases: list[str] = []
        attributes: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = format_signature(node)
            if "." in short_name:
                signature = f"{short_name.rsplit('.', 1)[0]}.{signature}"
            parameters = extract_parameters(node)
            returns = infer_returns(node, short_name)
        elif isinstance(node, ast.ClassDef):
            bases, attributes = class_metadata(node)
        qualified = f"{name}.{short_name}" if name else short_name
        symbol = Symbol(
            qualified_name=qualified,
            kind=kind,
            line=node.lineno,
            signature=signature,
            summary=infer_symbol_purpose(
                qualified, short_name, kind, ast.get_docstring(node)
            ),
            parameters=parameters,
            returns=returns,
            bases=bases,
            attributes=attributes,
        )
        module.symbols.append(symbol)
        caller = qualified
        for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
            target = ""
            if isinstance(call.func, ast.Name):
                if call.func.id in local_names:
                    target = f"{name}.{call.func.id}"
                elif call.func.id in aliases:
                    target = aliases[call.func.id]
            elif isinstance(call.func, ast.Attribute):
                if isinstance(call.func.value, ast.Name):
                    owner = call.func.value.id
                    if owner in {"self", "cls"} and "." in short_name:
                        target = f"{name}.{short_name.rsplit('.', 1)[0]}.{call.func.attr}"
                    elif owner in aliases:
                        target = f"{aliases[owner]}.{call.func.attr}"
            if target:
                module.calls.add((caller, target))
                symbol.calls.add(target)
    return module


def first_party_files() -> list[Path]:
    """Return every maintained Python file included in documentation coverage.
    
    Returns:
        list[Path]: Result produced by this operation.
    """
    return sorted(
        path
        for directory in FIRST_PARTY_DIRS
        for path in (ROOT / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def runtime_modules(modules: list[Module]) -> list[Module]:
    """Execute time modules for the module.
    
    Args:
        modules (list[Module]): Configuration or runtime value named modules.
    
    Returns:
        list[Module]: Result produced by this operation.
    """
    return [module for module in modules if module.path.relative_to(ROOT).parts[0] in RUNTIME_DIRS]


def shell_symbols(path: Path) -> list[tuple[str, int]]:
    """Extract shell function names and source lines from one launcher script.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
    
    Returns:
        list[tuple[str, int]]: Result produced by this operation.
    """
    symbols = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{", line)
        if match:
            symbols.append((match.group(1), line_number))
    return symbols


def role_for(path: str) -> str:
    """Classify a retained source path by its repository responsibility.
    
    Args:
        path (str): Filesystem path consumed by the operation.
    
    Returns:
        str: Result produced by this operation.
    """
    if path.startswith("./benchmark/devices/"):
        return "Device state application, verification, and restoration"
    if path.startswith("./benchmark/monitors/"):
        return "Telemetry sampling and interval statistics"
    if path == "./benchmark/data.py":
        return "CSV validation, median imputation, source-local windows, prepared batches"
    if path == "./benchmark/runner.py":
        return "Two-phase execution, timing boundaries, aggregation, persistence"
    if path == "./benchmark/config.py":
        return "Typed protocol and result schemas"
    if path.startswith("./models/"):
        return "Model adapter or lazy adapter factory"
    if path.startswith("./experiments/"):
        return "CLI entry point and device/profile/monitor selection"
    if path.startswith("./scripts/"):
        return "Environment setup, resume planning, validation, or documentation generation"
    if path.startswith("./tests/"):
        return "First-party regression test"
    if path.startswith("./utils/"):
        return "Active vendored upstream support or shared accuracy metrics"
    if path.startswith("./docs/"):
        return "Research and operations documentation"
    if path.endswith(".ipynb"):
        return "Analysis or publication notebook"
    if path.endswith(".sh"):
        return "Shell orchestration"
    return "Repository source or publication artifact"


def write_file_index(source_paths: list[str]) -> None:
    """Write file index for the module.
    
    Args:
        source_paths (list[str]): Configuration or runtime value named source_paths.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    lines = [
        "# Complete Retained-Source Index",
        "",
        "Generated from `docs/CODEBASE_SOURCE_TREE.txt`. Every retained source/documentation file appears exactly once.",
        "",
        "| Path | Role boundary |",
        "|---|---|",
    ]
    for path in source_paths:
        link = path.removeprefix("./")
        lines.append(f"| [`{path}`](../../{link}) | {role_for(path)} |")
    (OUTPUT / "FILE_INDEX.md").write_text("\n".join(lines) + "\n")


def write_symbol_index(modules: list[Module], shells: list[Path]) -> None:
    """Write symbol index for the module.
    
    Args:
        modules (list[Module]): Configuration or runtime value named modules.
        shells (list[Path]): Configuration or runtime value named shells.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    lines = [
        "# First-Party Symbol Index",
        "",
        "This is an AST-derived inventory. Dynamic calls and framework dispatch are explained in the curated workflow maps.",
        "",
        "| Symbol | Kind | Source | Purpose/docstring |",
        "|---|---|---|---|",
    ]
    for module in modules:
        relative = module.path.relative_to(ROOT)
        for symbol in module.symbols:
            lines.append(
                f"| `{symbol.qualified_name}` | {symbol.kind} | "
                f"[`{relative}:{symbol.line}`](../../{relative}#L{symbol.line}) | {symbol.summary} |"
            )
    for shell in shells:
        relative = shell.relative_to(ROOT)
        for name, line_number in shell_symbols(shell):
            lines.append(
                f"| `{relative}:{name}` | shell function | "
                f"[`{relative}:{line_number}`](../../{relative}#L{line_number}) | Shell orchestration helper. |"
            )
    (OUTPUT / "SYMBOL_INDEX.md").write_text("\n".join(lines) + "\n")


def parameter_compact(parameter: Parameter) -> str:
    """Format one parameter's name, type, and default for a compact mind-map node.
    
    Args:
        parameter (Parameter): Configuration or runtime value named parameter.
    
    Returns:
        str: Result produced by this operation.
    """
    default = "" if parameter.default == "required" else f" = {parameter.default}"
    return f"`{parameter.name}: {parameter.annotation}{default}`"


def write_api_reference(modules: list[Module], shells: list[Path]) -> None:
    """Write api reference for the module.
    
    Args:
        modules (list[Module]): Configuration or runtime value named modules.
        shells (list[Path]): Configuration or runtime value named shells.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    lines = [
        "# First-Party API Reference",
        "",
        "This reference is regenerated from the Python AST. Signatures, annotations, defaults, inheritance, instance fields, source lines, and static calls are source-derived. Purpose text uses source docstrings when available and a clearly conservative name/role inference otherwise. Dynamic framework dispatch is documented in the workflow and pseudocode documents.",
        "",
    ]
    for module in modules:
        relative = module.path.relative_to(ROOT)
        lines.extend(
            [
                f"## `{module.name or relative.stem}`",
                "",
                f"**Source:** [`{relative}`](../../{relative})",
                "",
                f"**Module responsibility:** {module.summary}",
                "",
            ]
        )
        if not module.symbols:
            lines.extend(["No public or private class/function declarations.", ""])
            continue
        for symbol in module.symbols:
            lines.extend(
                [
                    f"### `{symbol.signature}`",
                    "",
                    f"**Kind:** {symbol.kind}  ",
                    f"**Source:** [`{relative}:{symbol.line}`](../../{relative}#L{symbol.line})  ",
                    f"**Purpose:** {symbol.summary}",
                    "",
                ]
            )
            if symbol.kind == "class":
                lines.append(
                    "**Bases:** "
                    + (", ".join(f"`{base}`" for base in symbol.bases) if symbol.bases else "implicit `object`")
                )
                lines.append("")
                lines.append("**Declared/observed state fields:**")
                lines.append("")
                if symbol.attributes:
                    lines.extend(f"- `{attribute}`" for attribute in symbol.attributes)
                else:
                    lines.append("- No class or `self` fields are statically declared.")
                lines.append("")
            else:
                lines.extend(
                    [
                        "**Parameters:**",
                        "",
                        "| Name | Kind | Type | Default | Meaning |",
                        "|---|---|---|---|---|",
                    ]
                )
                if symbol.parameters:
                    for parameter in symbol.parameters:
                        lines.append(
                            f"| `{parameter.name}` | {parameter.kind} | `{parameter.annotation}` | "
                            f"`{parameter.default}` | {parameter.description} |"
                        )
                else:
                    lines.append("| _none_ | - | - | - | Callable takes no parameters. |")
                lines.extend(["", f"**Returns:** {symbol.returns}", ""])
            lines.append("**Statically resolved calls:**")
            lines.append("")
            if symbol.calls:
                lines.extend(f"- `{target}`" for target in sorted(symbol.calls))
            else:
                lines.append("- None resolved; calls may be dynamic, built-in, or absent.")
            lines.append("")

    lines.extend(
        [
            "## Shell Functions",
            "",
            "Shell helpers primarily communicate through launcher environment variables and process exit status; they do not have Python-style typed signatures.",
            "",
            "| Function | Source | Purpose |",
            "|---|---|---|",
        ]
    )
    for shell in shells:
        relative = shell.relative_to(ROOT)
        for name, line_number in shell_symbols(shell):
            purpose = words_from_identifier(name)
            lines.append(
                f"| `{name}()` | [`{relative}:{line_number}`](../../{relative}#L{line_number}) | "
                f"Perform the **{purpose}** launcher/setup step using shell state and explicit command exit codes. |"
            )
    (OUTPUT / "API_REFERENCE.md").write_text("\n".join(lines) + "\n")


def subsystem(path: Path) -> str:
    """Return the architecture subsystem assigned to a first-party module.
    
    Args:
        path (Path): Filesystem path consumed by the operation.
    
    Returns:
        str: Result produced by this operation.
    """
    relative = path.relative_to(ROOT)
    if relative.parts[0] == "benchmark" and len(relative.parts) > 1:
        if relative.parts[1] in {"devices", "monitors"}:
            return f"benchmark/{relative.parts[1]}"
        return "benchmark/core"
    return relative.parts[0]


def write_markmaps(modules: list[Module], source_paths: list[str], shells: list[Path]) -> None:
    """Write markmaps for the module.
    
    Args:
        modules (list[Module]): Configuration or runtime value named modules.
        source_paths (list[str]): Configuration or runtime value named source_paths.
        shells (list[Path]): Configuration or runtime value named shells.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    grouped: dict[str, list[Module]] = {}
    for module in modules:
        grouped.setdefault(subsystem(module.path), []).append(module)
    lines = [
        "---",
        "markmap:",
        "  colorFreezeLevel: 2",
        "  initialExpandLevel: 2",
        "  maxWidth: 320",
        "---",
        "# PowerBench Inference Codebase",
        "## User entry points",
        "- `run_unified_benchmark.sh`: device-aware two-phase production launcher",
        "- `smoke_test.sh`: all-model two-phase qualification through the production path",
        "- `evaluate_saved_predictions.ipynb`: accuracy from Phase-1 NPZ files",
        "- `plot_5090_publication_results.ipynb`: publication plots",
        "## Phase 1: deterministic prediction generation",
        "- Independent source CSV discovery and validation",
        "- Per-source global median imputation; observed mask retained",
        "- Source-local windows; no cross-file leakage; no shuffle",
        "- One model process and isolated environment",
        "- Save one NPZ per source; no telemetry or accuracy calculation",
        "## Phase 2: performance benchmark",
        "- Prebuild deterministic batch pool before timing",
        "- Apply/verify power mode and stabilize",
        "- Idle baseline, discarded warmups, measured repeats",
        "- 50 ms telemetry parallel to synchronous model calls",
        "- Mean, sample standard deviation, Student-t 95% CI",
        "- No predictions or raw telemetry persistence",
        "## Runtime subsystems",
    ]
    for group, items in sorted(grouped.items()):
        if group == "tests":
            continue
        lines.append(f"### {group}")
        for module in items:
            relative = module.path.relative_to(ROOT)
            lines.append(f"- `{relative}`")
            lines.append(f"  - {module.summary}")
            for symbol in module.symbols[:12]:
                lines.append(f"  - `{symbol.signature}`")
                lines.append(f"    - Purpose: {symbol.summary}")
                if symbol.parameters:
                    lines.append(
                        "    - Parameters: "
                        + ", ".join(parameter_compact(item) for item in symbol.parameters)
                    )
                if symbol.kind != "class":
                    lines.append(f"    - Returns: {symbol.returns}")
            if len(module.symbols) > 12:
                lines.append(f"  - {len(module.symbols) - 12} additional symbols in API_REFERENCE and symbol mind map")
    lines.extend(
        [
            "## Models",
            "- Adapter contract: `(B, context) -> (B, horizon), adapter-reported time`",
            "- Lazy factory prevents unused dependency imports",
            "- 13 registered models; TTM r2 and r3 share `.venv_ttm` but not outputs",
            "## Devices and telemetry",
            "- H100 / H200 / RTX 5090 / A6000: selected-index NVML profile and 50 ms monitor",
            "- Jetson Orin: MAXN, jetson_clocks, rails, thermals, OC invalidation",
            "- CPU/RPi: governor, RAPL/sysfs, RSS, thermal/throttle validation",
            "## Reliability",
            "- Fixed global seed and deterministic source/batch order",
            "- GPU/CPU memory guards and OOM/timeout ceilings",
            "- Prediction, benchmark, and supplemental model-boundary resume",
            "- Fail-closed result and prediction validation",
            "## Generated understanding package",
            "- Complete file index",
            "- Complete first-party API and symbol indexes",
            "- Abstract execution pseudocode with measured-region boundaries",
            "- Module import graph",
            "- Statically resolved call graph",
            "- Mermaid sequence and flow diagrams",
            "- D2 publication architecture",
        ]
    )
    (OUTPUT / "01_POWERBENCH_MINDMAP.md").write_text("\n".join(lines) + "\n")

    details = [
        "---",
        "markmap:",
        "  colorFreezeLevel: 2",
        "  initialExpandLevel: 2",
        "  maxWidth: 360",
        "---",
        "# PowerBench First-Party Symbols",
    ]
    for group, items in sorted(grouped.items()):
        details.append(f"## {group}")
        for module in items:
            details.append(f"### {module.name or module.path.stem}")
            details.append(f"- `{module.path.relative_to(ROOT)}`")
            for symbol in module.symbols:
                details.append(f"  - `{symbol.signature}` [{symbol.kind}, line {symbol.line}]")
                details.append(f"    - Purpose: {symbol.summary}")
                if symbol.kind == "class":
                    details.append(
                        "    - Bases: "
                        + (", ".join(f"`{base}`" for base in symbol.bases) if symbol.bases else "implicit `object`")
                    )
                    if symbol.attributes:
                        details.append(f"    - State fields ({len(symbol.attributes)})")
                        for attribute in symbol.attributes:
                            details.append(f"      - `{attribute}`")
                else:
                    details.append("    - Parameters")
                    if symbol.parameters:
                        for parameter in symbol.parameters:
                            details.append(f"      - {parameter_compact(parameter)}: {parameter.description}")
                    else:
                        details.append("      - None")
                    details.append(f"    - Returns: {symbol.returns}")
                details.append("    - Static calls")
                if symbol.calls:
                    for target in sorted(symbol.calls):
                        details.append(f"      - `{target}`")
                else:
                    details.append("      - None resolved")
    details.append("## Shell orchestration")
    for shell in shells:
        details.append(f"### {shell.relative_to(ROOT)}")
        for name, line_number in shell_symbols(shell):
            details.append(f"- `{name}()` [line {line_number}]")
            details.append(f"  - Purpose: perform the {words_from_identifier(name)} shell orchestration step")
            details.append("  - Inputs: launcher environment and shell arguments/state")
            details.append("  - Result: process state, emitted status, or command exit code")
    (OUTPUT / "02_SYMBOL_MINDMAP.md").write_text("\n".join(details) + "\n")


def write_graphviz(modules: list[Module]) -> None:
    """Write graphviz for the module.
    
    Args:
        modules (list[Module]): Configuration or runtime value named modules.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    runtime = runtime_modules(modules)
    known = {module.name for module in runtime if module.name}
    colors = {"benchmark": "#0E7490", "experiments": "#D97706", "models": "#15803D", "scripts": "#B91C1C"}
    lines = [
        "digraph imports {",
        'graph [rankdir=LR, bgcolor="#FAFAF7", pad=0.3, nodesep=0.35, ranksep=0.8, fontname="Helvetica"];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#334155", fontcolor="white"];',
        'edge [color="#64748B", arrowsize=0.7];',
    ]
    for module in runtime:
        if not module.name:
            continue
        group = module.name.split(".")[0]
        lines.append(f'"{module.name}" [fillcolor="{colors.get(group, "#475569")}"];')
    for module in runtime:
        for imported in module.imports:
            candidates = [name for name in known if imported == name or imported.startswith(name + ".")]
            if candidates:
                target = max(candidates, key=len)
                if target != module.name:
                    lines.append(f'"{module.name}" -> "{target}";')
    lines.append("}")
    header = lines[:4]
    statements = sorted(set(lines[4:-1]))
    (OUTPUT / "10_MODULE_DEPENDENCIES.dot").write_text(
        "\n".join([*header, *statements, "}"]) + "\n"
    )

    calls = sorted({edge for module in runtime for edge in module.calls})
    lines = [
        "digraph calls {",
        'graph [rankdir=LR, bgcolor="#FAFAF7", concentrate=true, pad=0.3, fontname="Helvetica"];',
        'node [shape=box, style="rounded,filled", fillcolor="#1E293B", fontcolor="white", fontsize=8, fontname="Helvetica"];',
        'edge [color="#94A3B8", arrowsize=0.55];',
    ]
    for caller, target in calls:
        if target.startswith(("benchmark.", "experiments.", "models.", "scripts.")):
            lines.append(f'"{caller}" -> "{target}";')
    lines.append("}")
    (OUTPUT / "11_STATIC_CALL_GRAPH.dot").write_text("\n".join(lines) + "\n")


def write_curated_diagrams() -> None:
    """Write curated diagrams for the module.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    diagrams = {
        "03_END_TO_END_FLOW.mmd": """flowchart TB
    subgraph setup[Setup]
      direction LR
      A[run_unified_benchmark.sh] --> B[Device config and host guards]
      B --> C[Per-model environment preflight]
    end
    subgraph phase1[Phase 1 - deterministic predictions]
      direction LR
      D[Prediction process] --> E[Independent CSV preprocessing]
      E --> F[Source-local windows]
      F --> G[Model adapter forecast]
      G --> H[Per-source NPZ predictions]
      H --> I[Prediction validation]
    end
    subgraph transition[Phase boundary]
      direction LR
      J[20 minute cooling] --> K[Clear accelerator and process state]
    end
    subgraph phase2[Phase 2 - performance benchmark]
      direction LR
      L[Benchmark process] --> M[Prepared deterministic batch pool]
      M --> N[Power mode verification and stabilization]
      N --> O[Idle baseline and warmups]
      O --> P[Measured repeats plus 50 ms telemetry]
      P --> Q[Repeat metrics]
      Q --> R[Mean, sample std, 95 percent t-CI]
      R --> S[Summary CSV, metadata, logs]
      S --> T[Fail-closed validation]
    end
    C --> D
    I --> J
    K --> L
""",
        "04_PREDICTION_SEQUENCE.mmd": """sequenceDiagram
    participant L as Launcher
    participant E as experiments.forecasting
    participant R as BenchmarkRunner
    participant D as benchmark.data
    participant F as models.factory
    participant M as Model adapter
    participant O as NPZ output
    L->>E: New isolated model interpreter, phase=prediction
    E->>R: Config plus device profile and monitor
    R->>D: Load matching CSV sources
    D-->>R: Validated contiguous SourceSeries
    R->>D: Build source-local windows
    R->>F: get_model(model, context, horizon)
    F-->>R: Lazy adapter instance
    loop Fixed source order, shuffle false
        R->>M: forecast(batch, horizon)
        M-->>R: Point forecast
        R->>O: Memory-map then save source NPZ
    end
    R-->>L: Complete marker with no telemetry or accuracy
""",
        "05_BENCHMARK_SEQUENCE.mmd": """sequenceDiagram
    participant L as Launcher
    participant R as BenchmarkRunner
    participant P as DeviceProfile
    participant D as PreparedBenchmarkStream
    participant M as Model adapter
    participant T as PowerMonitor and RSS sampler
    participant C as Summary CSV
    L->>R: phase=benchmark, one model process
    R->>P: setup, apply mode, verify, stabilize
    R->>D: Prebuild deterministic pool before timing
    R->>M: Load model and stabilize
    R->>T: Measure idle baseline
    loop Discarded warmup iterations
        loop Fixed number of preloaded batches
            R->>M: forecast zero-copy batch view
        end
    end
    loop Measured repeats
        R->>T: Start 50 ms telemetry and RSS
        R->>M: Synchronous forecasts over fixed batches
        R->>T: Stop and align to timestamps
        R->>R: Compute repeat metrics and validity
    end
    R->>C: Append mean, sample std, 95 percent t-CI
    R->>P: Restore device state
""",
        "06_DATA_PIPELINE.mmd": """flowchart TD
    A[DATA_DIR plus SOURCE_PATTERN] --> B[Stable recursive source discovery]
    B --> C[Read each CSV independently]
    C --> D[Detect target and time columns]
    D --> E[Parse and stable-sort timestamps]
    E --> F[Collapse duplicate timestamps by median]
    F --> G[Validate target and infer cadence]
    G --> H[Per-source global median imputation]
    H --> I[Retain original observed mask]
    I --> J[Split timestamp discontinuities]
    J --> K[Contiguous float32 arrays and cache]
    K --> L[Window each source independently]
    L --> M[Skip incomplete final window]
    M --> N[Phase 1 DataLoader, shuffle false]
    M --> O[Phase 2 preloaded deterministic pool]
""",
        "07_RESUME_AND_FAILURES.mmd": """flowchart TD
    A[Start or rerun] --> B{Phase}
    B -->|Prediction| C[Validate NPZ and completion marker per model]
    C --> D{First incomplete model?}
    D -->|Yes| E[Backup and remove only affected model and later prediction state]
    D -->|No| Z[Nothing pending]
    B -->|Benchmark| F[Validate summary rows and completion marker]
    F --> G{Base or supplemental?}
    G -->|Base| H[Restart first incomplete model boundary]
    G -->|Supplemental| I[Preserve all other batch sizes; replace only partial requested rows]
    H --> J{Runtime outcome}
    I --> J
    J -->|Success| K[Validate and continue]
    J -->|OOM or timeout| L[Record terminal ceiling; skip larger batch only for same context and horizon]
    J -->|Error, swap, throttle| M[Fail closed and stop model sequence]
""",
        "08_METRICS_PIPELINE.mmd": """flowchart LR
    A[Monotonic timed interval] --> B[Processed windows and points]
    B --> C[Throughput and latency]
    D[50 ms timestamped telemetry] --> E[Clip all channels; interpolate power boundaries]
    E --> F[Power mean, std, p50, p95, peak]
    E --> G[Trapezoidal energy integration]
    E --> H[GPU memory, utilization, temperature]
    I[Process tree RSS sampler] --> J[Peak CPU RAM]
    K[Idle baseline] --> L[Dynamic power and energy]
    C --> M[Per-repeat ResultRow]
    F --> M
    G --> M
    H --> M
    J --> M
    L --> M
    M --> N[Mean]
    M --> O[Sample standard deviation, ddof=1]
    M --> P[Two-sided 95 percent Student-t CI half-width]
    N --> Q[Batch summary row]
    O --> Q
    P --> Q
""",
        "09_BENCHMARK_LOOP.mmd": """flowchart TB
    A[Validated independent SourceSeries] --> B[Source-local MultiSourceWindowDataset]
    B --> C{power mode}
    C --> D{model}
    D --> E{context and horizon}
    E --> F{ascending batch size}
    F --> G[Memory guards and cooling]
    G --> H[Load model and stabilize]
    H --> I[Build PreparedBenchmarkStream before timing]
    I --> J[Measure idle baseline]
    J --> K{warmup plus measured iteration}
    K -->|warmup| L[Run all prepared batches without telemetry; discard outputs]
    L --> K
    K -->|measured repeat| M[Start process RSS sampler and hardware monitor]
    M --> N[Capture start timestamp and monotonic timer]
    N --> O{batch index below benchmark batch count}
    O -->|yes| P[Get zero-copy prepared batch view]
    P --> Q[Call model.forecast synchronously]
    Q --> R[Accumulate model-call time and processed counts]
    R --> O
    O -->|no| S[CUDA synchronize and capture end timestamp]
    S --> T[Stop monitor and RSS sampler]
    T --> U[Align timestamped samples to measured interval]
    U --> V[Compute repeat timing, throughput, energy, memory, utilization, and validity]
    V --> W[Keep ResultRow in memory]
    W --> K
    K -->|all repeats complete| X[Mean, sample std, and 95 percent Student-t CI]
    X --> Y[Append one batch-summary row]
    Y --> Z{OOM or timeout ceiling}
    Z -->|no| F
    Z -->|yes| AA[Skip only larger batches for this context and horizon]
    AA --> E

    classDef measured fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d,stroke-width:2px;
    class M,N,O,P,Q,R,S,T,U measured;
""",
    }
    for name, content in diagrams.items():
        (OUTPUT / name).write_text(content)

    d2 = """direction: down
classes: {
  phase: {style.fill: "#0E7490"; style.font-color: white}
  core: {style.fill: "#1E293B"; style.font-color: white}
  output: {style.fill: "#15803D"; style.font-color: white}
  guard: {style.fill: "#B45309"; style.font-color: white}
}
operator: "Research operator" {shape: person}
launcher: "Unified launcher\\nrun_unified_benchmark.sh" {class: core}
host: "Host controls" {
  device: "Device profile" {class: guard}
  swap: "Swap / idle / memory guards" {class: guard}
  env: "Per-model isolated environment" {class: guard}
}
prediction: "Phase 1: prediction" {
  data: "CSV -> median -> source-local windows" {class: phase}
  adapter: "Lazy model adapter" {class: phase}
  npz: "Per-source NPZ" {class: output}
}
benchmark: "Phase 2: benchmark" {
  pool: "Prepared deterministic batches" {class: phase}
  timing: "Warmups + measured repeats" {class: phase}
  telemetry: "50 ms monitor + RSS" {class: phase}
  aggregate: "Mean + std + 95% t-CI" {class: phase}
}
artifacts: "Publication artifacts" {
  summary: "Batch summary CSV" {class: output}
  metadata: "Dataset/model/software/hardware metadata" {class: output}
  logs: "Per-model phase logs" {class: output}
}
operator -> launcher
launcher -> host.env
launcher -> host.device
launcher -> host.swap
launcher -> prediction.data
prediction.data -> prediction.adapter
prediction.adapter -> prediction.npz
prediction.npz -> benchmark.pool: validated before benchmark
benchmark.pool -> benchmark.timing
host.device -> benchmark.telemetry
benchmark.timing -> benchmark.telemetry
benchmark.telemetry -> benchmark.aggregate
benchmark.aggregate -> artifacts.summary
launcher -> artifacts.metadata
launcher -> artifacts.logs
"""
    (OUTPUT / "12_PUBLICATION_ARCHITECTURE.d2").write_text(d2)


def write_readme(stats: dict[str, int]) -> None:
    """Write readme for the module.
    
    Args:
        stats (dict[str, int]): Configuration or runtime value named stats.
    
    Returns:
        None; the operation updates state, validates input, or persists output in place.
    """
    text = f"""# PowerBench Codebase Maps

This generated package explains the repository at four complementary levels.

1. Open `rendered/01_POWERBENCH_MINDMAP.html` for the collapsible Markmap learning view.
2. Open Mermaid SVGs for exact runtime, prediction, benchmark, data, resume, metrics, and inner-loop flows.
3. Open Graphviz SVGs for AST-derived module dependencies and statically resolved calls.
4. Open the D2 SVG for a publication-style system architecture.
5. Read `API_REFERENCE.md` for every first-party class/function/method and `../EXECUTION_PSEUDOCODE.md` for the abstract loop.

## Why These Formats

- **Markmap** turns the Markdown hierarchy into a collapsible learning map and remains useful as the repository grows.
- **Mermaid** expresses curated runtime and sequence semantics close to the documentation source.
- **Graphviz through Viz.js** renders deterministic AST-derived dependency and call graphs without requiring a host Graphviz installation.
- **D2** provides a compact, publication-oriented architecture view with explicit containers and styling.

No single visualization is treated as authoritative. The generated indexes provide completeness, the AST graphs provide structural evidence, and the curated diagrams explain dynamic behavior that static analysis cannot reliably infer.

## Completeness

- Retained source/documentation files indexed: **{stats['source_files']}**
- First-party Python files parsed: **{stats['python_files']}**
- First-party Python symbols indexed: **{stats['python_symbols']}**
- Shell functions indexed: **{stats['shell_symbols']}**
- Runtime import edges: **{stats['import_edges']}**
- Statically resolved call edges: **{stats['call_edges']}**

`FILE_INDEX.md`, `SYMBOL_INDEX.md`, `API_REFERENCE.md`, and `coverage.json` are the audit layer. Every retained file is classified in the file index. The API reference and mind maps derive definitions, parameters, defaults, return contracts, fields, and calls directly from source. AST symbol/import/call analysis is deliberately limited to benchmark-owned code under `benchmark/`, `experiments/`, `models/`, `scripts/`, and `tests/`; active upstream implementations under `utils/` are classified as vendored boundaries rather than represented as first-party logic. Curated diagrams summarize dynamic framework dispatch.

## Regenerate

```bash
python3 scripts/generate_codebase_maps.py
bash scripts/render_codebase_maps.sh
```

The renderer pins Markmap, Mermaid, Viz.js/Graphviz, and D2 versions and writes only under `docs/codebase_maps/rendered` and `.codebase_map_tools`.
"""
    (OUTPUT / "README.md").write_text(text)


def main() -> int:
    """Parse command-line arguments and execute the the module workflow.
    
    Returns:
        int: Computed scalar described by this callable.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    modules = [scan_module(path) for path in first_party_files()]
    source_paths = [
        line.strip()
        for line in (ROOT / "docs" / "CODEBASE_SOURCE_TREE.txt").read_text().splitlines()
        if line.strip()
    ]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("CODEBASE_SOURCE_TREE.txt contains duplicate paths")
    missing_sources = [path for path in source_paths if not (ROOT / path.removeprefix("./")).exists()]
    tracked_paths = set(source_paths)
    untracked_first_party = [
        str(path.relative_to(ROOT))
        for path in first_party_files()
        if f"./{path.relative_to(ROOT)}" not in tracked_paths
    ]
    if untracked_first_party:
        raise ValueError(f"First-party Python files are absent from the source inventory: {untracked_first_party}")
    shells = [ROOT / "run_unified_benchmark.sh", ROOT / "smoke_test.sh", *sorted((ROOT / "scripts").glob("*.sh"))]
    write_file_index(source_paths)
    write_symbol_index(modules, shells)
    write_api_reference(modules, shells)
    write_markmaps(modules, source_paths, shells)
    write_graphviz(modules)
    write_curated_diagrams()

    runtime = runtime_modules(modules)
    known = {module.name for module in runtime}
    import_edges = {
        (module.name, imported)
        for module in runtime
        for imported in module.imports
        if any(imported == name or imported.startswith(name + ".") for name in known)
    }
    stats = {
        "source_files": len(source_paths),
        "python_files": len(modules),
        "python_symbols": sum(len(module.symbols) for module in modules),
        "shell_symbols": sum(len(shell_symbols(path)) for path in shells),
        "import_edges": len(import_edges),
        "call_edges": sum(len(module.calls) for module in runtime),
    }
    coverage = {
        "schema_version": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stats": stats,
        "first_party_python_files": [str(module.path.relative_to(ROOT)) for module in modules],
        "missing_retained_paths_on_this_host": missing_sources,
        "unparsed_first_party_python_files": [],
        "retained_source_paths": source_paths,
    }
    (OUTPUT / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    write_readme(stats)
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
