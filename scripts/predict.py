#!/usr/bin/env python3
"""
ABodyBuilder3 Command Line Interface

Predict antibody 3D structures from heavy and light chain sequences.
Supports single sequences, FASTA files, and CSV batch processing.

Examples:
    # Single antibody from command line
    predict.py predict --heavy "EVQL..." --light "DIQM..." -o antibody.pdb
    
    # Batch processing from CSV (recommended for multiple antibodies)
    predict.py predict --csv antibodies.csv -o output_dir/
    
    # Using the Language model (highest accuracy, requires ProtT5)
    predict.py predict --csv input.csv -o output/ --model language
    
    # Chemical preparation, optionally followed by restrained minimisation
    predict.py predict --csv input.csv -o output/ --add-hydrogens
    predict.py predict --csv input.csv -o output/ --relax restrained

Input Formats:
    CSV: Must have columns 'name', 'heavy', 'light' (or 'VH', 'VL')
    FASTA: Pairs sequences by header keywords (_heavy/_light, _VH/_VL, _H/_L)
           or by position (alternating heavy/light)

Models:
    base     - Fast baseline model (~60 antibodies/min)
    plddt    - Base + pLDDT confidence scores (~30/min) [default]
    language - ProtT5 embeddings for best CDR accuracy (~4/min)
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import csv
import time
from enum import Enum

# Reject the retired flag before importing/loading the model runtime.
if any(arg in ("--relaxation", "-r") or arg.startswith("--relaxation=") for arg in sys.argv[1:]):
    raise SystemExit("Error: --relaxation/-r has been removed. It previously only added hydrogens. "
                     "Use --add-hydrogens for preparation or --relax restrained for genuine energy minimisation.")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import ml_collections
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Suppress warnings
warnings.filterwarnings("ignore")

# Register safe globals for PyTorch 2.10+
torch.serialization.add_safe_globals([ml_collections.ConfigDict])

from abodybuilder3.utils import string_to_input, output_to_pdb, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3


class OutputFormat(str, Enum):
    """Output file format."""
    pdb = "pdb"
    cif = "cif"
    json = "json"


HELP_EPILOG = """
For more information, run: predict.py info

CSV Format Example:
    name,heavy,light
    trastuzumab,EVQLVESGG...,DIQMTQSPS...
    adalimumab,EVQLVESGG...,DIQMTQSPS...
"""

app = typer.Typer(
    help="ABodyBuilder3: Predict antibody 3D structures from sequences.",
    epilog=HELP_EPILOG,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Global verbosity flag
VERBOSE = True


def log(message: str, level: str = "info"):
    """Log message based on verbosity level."""
    if not VERBOSE and level != "error":
        return
    if level == "error":
        console.print(f"[red]{message}[/red]")
    elif level == "warning":
        console.print(f"[yellow]{message}[/yellow]")
    else:
        console.print(message)


def get_optimal_workers() -> int:
    """Determine optimal number of workers based on benchmarks.
    
    Benchmarks show 8 workers is optimal for DGX Spark (751 antibodies/min).
    Performance drops at 10+ workers due to GPU contention.
    """
    return 8


def checkpoint_path(model_type: str, checkpoint: Optional[Path] = None):
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Custom checkpoint not found: {checkpoint}')
        path = checkpoint
    else:
        model_paths = {
            "plddt": "output/plddt-loss/best_second_stage.ckpt",
            "base": "output/base-loss/best_second_stage.ckpt",
            "language": "output/language-loss/best_second_stage.ckpt",
        }
        
        path = None
        for base in ["/opt/abodybuilder3", "."]:
            candidate = Path(base) / model_paths.get(model_type, model_paths["plddt"])
            if candidate.exists():
                path = candidate
                break
        
        if not path:
            raise FileNotFoundError(f"Model checkpoint not found for {model_type}")
    
    return Path(path)


def load_model(model_type: str, device: torch.device, checkpoint: Optional[Path] = None):
    """Load the specified model and retain its exact checkpoint identity."""
    path = checkpoint_path(model_type, checkpoint)
    module = LitABB3.load_from_checkpoint(str(path))
    model = module.model
    model.to(device)
    model.eval()
    from abodybuilder3.aot_runtime import sha256
    model._deployment_checkpoint_sha256 = sha256(path)
    return model


# Global ProtT5 model (lazy loaded)
_prott5_model = None
_prott5_device = None


def load_prott5(device: torch.device):
    """Load ProtT5 model for embedding generation (lazy load)."""
    global _prott5_model, _prott5_device
    
    if _prott5_model is not None and _prott5_device == device:
        return _prott5_model
    
    log("[dim]Loading ProtT5 model for language embeddings...[/dim]")
    from abodybuilder3.language.model import ProtT5
    
    _prott5_model = ProtT5(device_map=str(device) if device.type == 'cuda' else 'cpu')
    _prott5_device = device
    return _prott5_model


def prepare_input_with_embeddings(heavy: str, light: str, prott5_model, device: torch.device) -> Dict[str, Any]:
    """Prepare model input with ProtT5 embeddings for Language model."""
    from abodybuilder3.dataloader import ABDataset
    from abodybuilder3.openfold.np.residue_constants import restype_order_with_x
    
    # Build basic input
    aatype = []
    is_heavy = []
    for character in heavy:
        is_heavy.append(1)
        aatype.append(restype_order_with_x[character])
    for character in light:
        is_heavy.append(0)
        aatype.append(restype_order_with_x[character])
    is_heavy = torch.tensor(is_heavy)
    aatype = torch.tensor(aatype)
    residue_index = torch.cat(
        (torch.arange(len(heavy)), torch.arange(len(light)) + 500)
    )
    
    # Generate ProtT5 embeddings
    embeddings = prott5_model.get_embeddings([heavy], [light])
    plm_embedding = embeddings[0]  # (seq_len, 1024)
    
    # Build model input with PLM embeddings
    model_input = {
        "is_heavy": is_heavy,
        "aatype": aatype,
        "residue_index": residue_index,
        "plm_embedding": plm_embedding,
    }
    
    # Generate pair features
    pair = residue_index[None] - residue_index[:, None]
    pair = pair.clamp(-64, 64) + 64
    pair = torch.nn.functional.one_hot(pair, 2 * 64 + 1)
    
    # Add edge chain features
    is_heavy_tensor = model_input["is_heavy"]
    is_heavy_edge = 2 * is_heavy_tensor.outer(is_heavy_tensor) + (
        (1 - is_heavy_tensor).outer(1 - is_heavy_tensor)
    )
    is_heavy_edge = torch.nn.functional.one_hot(is_heavy_edge.long())
    pair = torch.cat((is_heavy_edge, pair), dim=-1)
    
    model_input["single"] = plm_embedding.float().unsqueeze(0)
    model_input["pair"] = pair.float().unsqueeze(0)
    
    model_input = {k: v.to(device) for k, v in model_input.items()}
    return model_input


def prepare_input(heavy: str, light: str) -> Dict[str, Any]:
    """Prepare model input (CPU-bound, can be parallelized)."""
    return string_to_input(heavy=heavy, light=light)


def run_inference(model, ab_input: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Run model inference."""
    ab_input_batch = {
        key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
        for key, value in ab_input.items()
    }
    
    with torch.no_grad():
        output = model(ab_input_batch, ab_input_batch["aatype"])
        output = add_atom37_to_output(output, ab_input["aatype"].to(device))
    
    return output


def extract_plddt_scores(output: Dict[str, Any]) -> List[float]:
    """Extract pLDDT scores from model output."""
    if "plddt" in output:
        plddt = output["plddt"]
        if hasattr(plddt, 'cpu'):
            plddt = plddt.cpu().numpy()
        # Flatten any nested arrays
        import numpy as np
        plddt_flat = np.array(plddt).flatten()
        return [float(x) for x in plddt_flat]
    return []


def generate_pdb(output: Dict[str, Any], ab_input: Dict[str, Any]) -> str:
    """Generate PDB string from model output."""
    # Formatting GPU scalar coordinates causes thousands of synchronisations.
    # Keep model computation unchanged and transfer the compact result once.
    cpu_output = dict(output)
    coords = cpu_output['atom37']
    cpu_output['atom37'] = coords.detach().cpu().numpy() if torch.is_tensor(coords) else coords
    mask = cpu_output['atom37_atom_exists']
    cpu_output['atom37_atom_exists'] = mask.detach().cpu() if torch.is_tensor(mask) else torch.as_tensor(mask)
    return output_to_pdb(cpu_output, ab_input)


def generate_cif(output: Dict[str, Any], ab_input: Dict[str, Any], name: str) -> str:
    """Generate mmCIF string from model output."""
    # Basic CIF generation (simplified)
    pdb_string = generate_pdb(output, ab_input)
    # Convert PDB to CIF format (basic conversion)
    lines = ["data_" + name, "#", "_entry.id " + name, "#"]
    lines.append("loop_")
    lines.append("_atom_site.group_PDB")
    lines.append("_atom_site.id")
    lines.append("_atom_site.type_symbol")
    lines.append("_atom_site.label_atom_id")
    lines.append("_atom_site.label_comp_id")
    lines.append("_atom_site.label_asym_id")
    lines.append("_atom_site.label_seq_id")
    lines.append("_atom_site.Cartn_x")
    lines.append("_atom_site.Cartn_y")
    lines.append("_atom_site.Cartn_z")
    
    atom_id = 1
    for line in pdb_string.split("\n"):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            group = line[0:6].strip()
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21:22].strip()
            res_seq = line[22:26].strip()
            x = line[30:38].strip()
            y = line[38:46].strip()
            z = line[46:54].strip()
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            
            lines.append(f"{group} {atom_id} {element} {atom_name} {res_name} {chain} {res_seq} {x} {y} {z}")
            atom_id += 1
    
    lines.append("#")
    return "\n".join(lines)


def generate_json_output(
    name: str,
    heavy: str,
    light: str,
    output: Dict[str, Any],
    ab_input: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate JSON output with structure and metadata."""
    plddt_scores = extract_plddt_scores(output)
    mean_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0
    
    return {
        "name": name,
        "heavy_chain": heavy,
        "light_chain": light,
        "pdb": generate_pdb(output, ab_input),
        "plddt_scores": plddt_scores,
        "mean_plddt": mean_plddt,
        "num_residues": len(heavy) + len(light),
    }


def write_prepared_outputs(raw, output_path, options):
    """Preserve raw output and save requested preparation separately."""
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from structure_preparation import PreparationError, prepare_structure
    record_path = output_path.with_suffix(".preparation.json")
    try:
        prepared, relaxed, record = prepare_structure(raw, options)
        output_path.with_suffix(".prepared.pdb").write_text(prepared)
        if relaxed is not None:
            output_path.with_suffix(".relaxed.pdb").write_text(relaxed)
        for warning in record["warnings"]:
            log(f"{output_path.name}: {warning}", "error")
    except PreparationError as exc:
        record_path.write_text(json.dumps(exc.report, indent=2) + "\n")
        raise
    record_path.write_text(json.dumps(record, indent=2) + "\n")


def process_single_antibody(
    name: str,
    heavy: str,
    light: str,
    model,
    device: torch.device,
    output_format: OutputFormat = OutputFormat.pdb,
    confidence_threshold: Optional[float] = None,
    prott5_model = None,
) -> Tuple[str, str, Optional[float]]:
    """Process a single antibody end-to-end. Returns (name, output_string, mean_plddt)."""
    
    # Prepare input (with embeddings for language model)
    if prott5_model is not None:
        ab_input = prepare_input_with_embeddings(heavy, light, prott5_model, device)
    else:
        ab_input = prepare_input(heavy, light)
    
    output = run_inference(model, ab_input, device)
    
    # Extract pLDDT scores
    plddt_scores = extract_plddt_scores(output)
    mean_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else None
    
    # Check confidence threshold
    if confidence_threshold is not None and mean_plddt is not None:
        if mean_plddt < confidence_threshold:
            return name, None, mean_plddt
    
    # Generate output in requested format
    if output_format == OutputFormat.json:
        output_string = json.dumps(generate_json_output(name, heavy, light, output, ab_input), indent=2)
    elif output_format == OutputFormat.cif:
        output_string = generate_cif(output, ab_input, name)
    else:
        output_string = generate_pdb(output, ab_input)
    
    return name, output_string, mean_plddt


# Worker state for multiprocessing
_worker_model = None
_worker_device = None
_worker_prott5 = None
_worker_config = None


def _init_worker(model_type: str, checkpoint_path: Optional[str], use_language: bool, precision: str = 'strict'):
    """Initialize worker process with its own model instance."""
    global _worker_model, _worker_device, _worker_prott5, _worker_config
    
    # Suppress warnings in worker
    import warnings
    warnings.filterwarnings("ignore")
    
    # Register safe globals
    torch.serialization.add_safe_globals([ml_collections.ConfigDict])
    from abodybuilder3.aot_runtime import configure_precision
    configure_precision(precision)
    
    # Setup device
    _worker_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    _worker_model = load_model(model_type, _worker_device, Path(checkpoint_path) if checkpoint_path else None)
    
    # Load ProtT5 for language model
    if use_language:
        _worker_prott5 = load_prott5(_worker_device)
    
    _worker_config = {
        "model_type": model_type,
        "use_language": use_language,
    }


def _worker_process_antibody(args: Tuple) -> Tuple[str, Optional[str], Optional[float]]:
    """Process a single antibody in worker process."""
    name, heavy, light, output_format, confidence_threshold = args
    
    global _worker_model, _worker_device, _worker_prott5
    
    return process_single_antibody(
        name, heavy, light, _worker_model, _worker_device,
        output_format=output_format, confidence_threshold=confidence_threshold,
        prott5_model=_worker_prott5,
    )


def parse_fasta(fasta_path: Path) -> List[Tuple[str, str, str]]:
    """Parse FASTA file with paired heavy/light chains."""
    entries = []
    current_name = None
    current_seq = []
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    entries.append((current_name, "".join(current_seq)))
                current_name = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_name:
            entries.append((current_name, "".join(current_seq)))
    
    def get_chain_type(name: str) -> str:
        name_lower = name.lower()
        if any(x in name_lower for x in ["_heavy", "_vh", "_h", ":h", "-h"]):
            return "heavy"
        elif any(x in name_lower for x in ["_light", "_vl", "_l", ":l", "-l"]):
            return "light"
        return "unknown"
    
    def get_base_name(name: str) -> str:
        for suffix in ["_heavy", "_light", "_VH", "_VL", "_H", "_L", ":H", ":L", "-H", "-L"]:
            if name.endswith(suffix) or name.lower().endswith(suffix.lower()):
                return name[:-len(suffix)]
        return name
    
    heavy_chains = {}
    light_chains = {}
    unknown_chains = []
    
    for name, seq in entries:
        chain_type = get_chain_type(name)
        base_name = get_base_name(name)
        
        if chain_type == "heavy":
            heavy_chains[base_name] = seq
        elif chain_type == "light":
            light_chains[base_name] = seq
        else:
            unknown_chains.append((name, seq))
    
    antibodies = []
    for base_name in heavy_chains:
        if base_name in light_chains:
            antibodies.append((base_name, heavy_chains[base_name], light_chains[base_name]))
    
    if not antibodies and unknown_chains:
        for i in range(0, len(unknown_chains), 2):
            if i + 1 < len(unknown_chains):
                heavy_name, heavy_seq = unknown_chains[i]
                light_name, light_seq = unknown_chains[i + 1]
                name = get_base_name(heavy_name)
                antibodies.append((name, heavy_seq, light_seq))
    
    return antibodies


def parse_csv(csv_path: Path) -> List[Tuple[str, str, str]]:
    """Parse CSV file with columns: name, heavy, light."""
    antibodies = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", row.get("id", f"antibody_{len(antibodies)+1}"))
            heavy = row.get("heavy", row.get("heavy_chain", row.get("VH", "")))
            light = row.get("light", row.get("light_chain", row.get("VL", "")))
            if heavy and light:
                antibodies.append((name, heavy, light))
    return antibodies


def run_batched_pipeline(antibodies, model, device, output, output_format,
                         confidence_threshold, preparation_options, max_batch_size,
                         memory_gib, cpu_workers, report_path, unique_name, progress, task):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from batch_pipeline import BatchPolicy, plan_batches, run_pipeline
    from abodybuilder3.batched_inference import prepare_cpu_batch, infer_prepared_batch
    import threading
    named = [(unique_name(name), h, l) for name, h, l in antibodies]
    # Calibrated conservative estimate from the 227-231-residue FP32 screen.
    # OOM retries remain necessary: this is not a hard allocator/physical limit.
    pair_budget = max(1, int((memory_gib - .25) * 2**30 / 3000))
    policy = BatchPolicy(max_batch_size, pair_budget)
    stats = {"completed": False, "max_batch_size": max_batch_size,
             "memory_target_gib": memory_gib, "pair_budget": pair_budget,
             "batches": [], "oom_retries": [], "saved": 0, "skipped": 0}
    stats['deployment'] = getattr(model, '_deployment_stats', {'execution': 'eager'})
    lock = threading.Lock()
    prepared_paths = []
    is_dir = len(named) > 1 or output.is_dir()
    ext = {"pdb": ".pdb", "cif": ".cif", "json": ".json"}[output_format.value]
    if is_dir:
        output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def prepare(items):
        return prepare_cpu_batch([(h, l) for _, h, l in items], compact=True)

    def infer(items, prepared):
        try:
            outputs, inputs = infer_prepared_batch(model, prepared, device)
        except torch.cuda.OutOfMemoryError:
            if len(items) == 1:
                raise
        else:
            stats["batches"].append({"size": len(items), "max_length": max(len(h)+len(l) for _,h,l in items)})
            return list(zip(items, outputs, inputs))
        # Exit the exception handler first so its traceback releases GPU arrays.
        stats["oom_retries"].append([x[0] for x in items])
        log(f"CUDA memory exhausted for {len(items)} antibodies; splitting batch.", "warning")
        torch.cuda.empty_cache()
        middle = len(items)//2
        return infer(items[:middle], prepare(items[:middle])) + infer(items[middle:], prepare(items[middle:]))

    def write(records):
        for (name, heavy, light), result, inp in records:
            scores = extract_plddt_scores(result)
            mean = sum(scores)/len(scores) if scores else None
            if confidence_threshold is not None and mean is not None and mean < confidence_threshold:
                with lock:
                    stats["skipped"] += 1
                    progress.update(task, advance=1)
                continue
            if output_format == OutputFormat.json:
                text = json.dumps(generate_json_output(name, heavy, light, result, inp), indent=2)
            elif output_format == OutputFormat.cif:
                text = generate_cif(result, inp, name)
            else:
                text = generate_pdb(result, inp)
            path = output / (name+ext) if is_dir else (output if str(output).endswith(ext) else output.with_suffix(ext))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            with lock:
                stats["saved"] += 1
                if preparation_options is not None:
                    prepared_paths.append(path)
                progress.update(task, advance=1)

    try:
        run_pipeline(plan_batches(named, policy), prepare, infer, write,
                     prefetch=cpu_workers, writers=cpu_workers)
        # Keep OpenMM CUDA work separate from the inference engine, as on main.
        for path in prepared_paths:
            write_prepared_outputs(path.read_text(), path, preparation_options)
        stats["completed"] = True
    finally:
        stats["processing_seconds"] = time.perf_counter()-start
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(stats, indent=2)+"\n")
    log(f"Saved {stats['saved']} raw predictions; {stats['skipped']} filtered; "
        f"{len(stats['batches'])} GPU batches in {stats['processing_seconds']:.2f}s.")


@app.command()
def predict(
    # Input options
    heavy: Optional[str] = typer.Option(
        None, "--heavy", "-H",
        help="Heavy chain amino acid sequence (VH). Use with --light for single antibody."
    ),
    light: Optional[str] = typer.Option(
        None, "--light", "-L", 
        help="Light chain amino acid sequence (VL). Use with --heavy for single antibody."
    ),
    name: str = typer.Option(
        "antibody", "--name", "-n",
        help="Name for the antibody. Used in output filename when using --heavy/--light."
    ),
    fasta: Optional[Path] = typer.Option(
        None, "--fasta", "-f",
        help="FASTA file with paired heavy/light chains. Headers should contain _heavy/_light or _VH/_VL suffixes."
    ),
    csv_file: Optional[Path] = typer.Option(
        None, "--csv", "-c",
        help="CSV file with columns: name, heavy, light (or VH, VL). Recommended for batch processing."
    ),
    
    # Output options
    output: Path = typer.Option(
        "output.pdb", "--output", "-o",
        help="Output file or directory. Use trailing / for directory output (e.g., output/)."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.pdb, "--format",
        help="Output format: 'pdb' (default), 'cif' (mmCIF), or 'json' (includes pLDDT scores)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="Overwrite existing output files without prompting."
    ),
    
    # Model options
    model_type: str = typer.Option(
        "plddt", "--model", "-m",
        help="Model: 'plddt' (default, with confidence), 'base' (fastest), 'language' (best accuracy, uses ProtT5)."
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint",
        help="Path to custom model checkpoint. Overrides --model selection."
    ),
    device_str: str = typer.Option(
        "cuda", "--device", "-d",
        help="Compute device: 'cuda' (GPU, default) or 'cpu'."
    ),
    execution: str = typer.Option(
        "auto", "--execution", help="auto: saved AOT for released CUDA base/plddt; eager for other supported configurations. aot requires matching packages."
    ),
    precision: str = typer.Option(
        "strict", "--precision", help="strict (default): FP32 without TF32; tf32: explicitly permit TF32 with eager execution."
    ),
    aot_root: Path = typer.Option(
        "/opt/abodybuilder3/aot_packages", "--aot-root", help="Directory containing the matching saved AOT manifest and executables."
    ),
    pdb_writer: str = typer.Option(
        "native", "--pdb-writer", help="native (default) or python reference. Both retain the existing PDB fields."
    ),
    
    # Processing options
    add_hydrogens: bool = typer.Option(
        False, "--add-hydrogens",
        help="Save a chemically prepared PDB alongside the unchanged raw prediction."
    ),
    hydrogen_method: Optional[str] = typer.Option(
        None, "--hydrogen-method", help="Preparation: openmm (default) or reduce then OpenMM protonation."
    ),
    ph: Optional[float] = typer.Option(
        None, "--ph", help="Final OpenMM preparation pH, 0-14 (default 7)."
    ),
    sidechain_flips: bool = typer.Option(
        False, "--sidechain-flips", help="Allow Reduce to change Asn/Gln/His side-chain orientations."
    ),
    relax: str = typer.Option(
        "none", "--relax", help="none (default) or restrained: prepare and energy-minimise with OpenMM."
    ),
    relax_restraint: float = typer.Option(
        1000., "--relax-restraint", help="Heavy-atom positional restraint, kJ/mol/nm^2."
    ),
    relax_tolerance: float = typer.Option(
        10., "--relax-tolerance", help="Minimisation RMS force tolerance, kJ/mol/nm."
    ),
    relax_max_iterations: int = typer.Option(
        1000, "--relax-max-iterations", help="Positive finite minimisation iteration budget per stage."
    ),
    relax_platform: str = typer.Option(
        "cpu", "--relax-platform", help="Minimisation platform: cpu (default) or cuda. No fallback."
    ),
    confidence_threshold: Optional[float] = typer.Option(
        None, "--confidence-threshold", "-t",
        help="Skip antibodies with mean pLDDT below this threshold (0-100). Only works with plddt model."
    ),
    workers: str = typer.Option(
        "auto", "--workers", "-w",
        help="Parallel workers: 'auto' (detects optimal), '1' (sequential), or integer. Affects I/O, not GPU."
    ),
    batch_size: str = typer.Option(
        "auto", "--batch-size", "-b",
        help="GPU batch size: auto (16 for up to 128 inputs, otherwise 32), a positive integer, or 1 for workers. Language uses 1."
    ),
    batch_memory_gib: float = typer.Option(
        6., "--batch-memory-gib", min=0.5,
        help="Estimated inference allocation target, not a hard total-memory limit."
    ),
    cpu_workers: int = typer.Option(
        2, "--cpu-workers", min=1, max=8,
        help="CPU preparation/output workers per stage; bounded queues."
    ),
    batch_report: Optional[Path] = typer.Option(
        None, "--batch-report", help="Save batch sizes, OOM retries and completion metadata as JSON."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s",
        help="Random seed for reproducible predictions."
    ),
    
    # Output verbosity
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress progress output. Only show errors."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed processing information."
    ),
):
    """
    Predict 3D structure of antibody from heavy and light chain sequences.
    
    Requires either --heavy/--light, --fasta, or --csv input.
    """
    global VERBOSE
    VERBOSE = not quiet
    from abodybuilder3.aot_runtime import configure_precision
    if execution not in ('auto', 'aot', 'eager'):
        raise typer.BadParameter('--execution must be auto, aot or eager.')
    if precision not in ('strict', 'tf32'):
        raise typer.BadParameter('--precision must be strict or tf32.')
    if model_type not in ('base', 'plddt', 'language'):
        raise typer.BadParameter('--model must be base, plddt or language.')
    if pdb_writer not in ('native', 'python'):
        raise typer.BadParameter('--pdb-writer must be native or python.')
    os.environ['ABB3_PDB_WRITER'] = pdb_writer
    configure_precision(precision)
    if checkpoint is not None and not checkpoint.is_file():
        raise typer.BadParameter(f'Custom checkpoint not found: {checkpoint}')
    import math
    if not math.isfinite(batch_memory_gib):
        raise typer.BadParameter("--batch-memory-gib must be finite.")
    if batch_size == "auto":
        max_batch_size = 1 if model_type == "language" else 32
    else:
        try:
            max_batch_size = int(batch_size)
        except ValueError:
            raise typer.BadParameter("--batch-size must be auto or a positive integer.")
        if max_batch_size < 1:
            raise typer.BadParameter("--batch-size must be positive.")
    use_batching = max_batch_size > 1
    if use_batching and model_type == "language":
        raise typer.BadParameter("Language batching is not qualified; use --batch-size 1 --workers 1.")
    if use_batching and workers not in ("auto", "1"):
        raise typer.BadParameter("GPU batching uses one model process; use --workers auto or --workers 1.")
    preparation_options = None
    if add_hydrogens or relax != "none":
        if output_format != OutputFormat.pdb:
            raise typer.BadParameter("Preparation/relaxation currently requires --format pdb; other formats are never silently ignored.")
        script_dir = str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from structure_preparation import Options
        preparation_options = Options(
            hydrogen_method=hydrogen_method or "openmm", ph=ph,
            sidechain_flips=sidechain_flips, relax=relax,
            restraint_k=relax_restraint, tolerance=relax_tolerance,
            max_iterations=relax_max_iterations, seed=seed if seed is not None else 0,
            relax_platform=relax_platform.upper(),
        )
        try:
            preparation_options.validate()
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif hydrogen_method is not None or ph is not None or sidechain_flips or relax_restraint != 1000. or relax_tolerance != 10. or relax_max_iterations != 1000 or relax_platform != "cpu":
        raise typer.BadParameter("Preparation settings require --add-hydrogens or --relax restrained.")
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Determine input mode
    antibodies = []
    
    if heavy and light:
        antibodies = [(name, heavy, light)]
    elif fasta:
        if not fasta.exists():
            log(f"Error: FASTA file not found: {fasta}", "error")
            raise typer.Exit(1)
        antibodies = parse_fasta(fasta)
    elif csv_file:
        if not csv_file.exists():
            log(f"Error: CSV file not found: {csv_file}", "error")
            raise typer.Exit(1)
        antibodies = parse_csv(csv_file)
    else:
        log("Error: Provide --heavy/--light, --fasta, or --csv input", "error")
        raise typer.Exit(1)
    
    if not antibodies:
        log("Error: No valid antibody sequences found", "error")
        raise typer.Exit(1)
    
    if batch_size == "auto" and use_batching and len(antibodies) <= 128:
        max_batch_size = 16

    # Check output exists
    if len(antibodies) > 1:
        if output.exists() and not output.is_dir():
            log("Error: Output must be a directory for multiple antibodies", "error")
            raise typer.Exit(1)
    else:
        # For single antibody: if output is dir, that's fine; only block if it's an existing file
        if output.exists() and output.is_file() and not overwrite:
            log(f"Error: Output file exists: {output} (use --overwrite)", "error")
            raise typer.Exit(1)
    
    # Setup device
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    eligible = model_type in ('base', 'plddt') and device.type == 'cuda' and checkpoint is None and precision == 'strict'
    selected_execution = ('aot' if eligible else 'eager') if execution == 'auto' else execution
    if selected_execution == 'aot' and not eligible:
        raise typer.BadParameter('AOT requires a released base/plddt checkpoint, CUDA and --precision strict. Use --execution eager for this configuration.')
    if selected_execution == 'aot':
        use_batching = True  # The saved batch-1 domain uses the same compact path.
        if workers not in ('auto', '1'):
            raise typer.BadParameter('AOT uses one model process; use --workers auto or 1.')
    if execution == 'auto' and selected_execution == 'eager':
        log(f'Execution: eager; model={model_type}, device={device}, precision={precision}, custom_checkpoint={checkpoint is not None}.', 'warning')
    
    # Determine number of workers
    if workers == "auto":
        num_workers = get_optimal_workers()
    else:
        try:
            num_workers = int(workers)
        except ValueError:
            raise typer.BadParameter("--workers must be auto or a positive integer.")
    if num_workers < 1:
        raise typer.BadParameter("--workers must be positive.")
    if use_batching or (model_type == "language" and workers == "auto"):
        num_workers = 1
    if use_batching:
        # Avoid native thread oversubscription across CPU pipeline stages.
        torch.set_num_threads(1)
    
    if len(antibodies) <= 2:
        num_workers = 1
    
    log(f"[bold]ABodyBuilder3[/bold] - {model_type} model on {device}")
    log(f"Processing {len(antibodies)} antibody(ies) with {num_workers} worker(s)...")
    if preparation_options is not None:
        log(f"[dim]Separate preparation: {preparation_options.hydrogen_method}; relaxation: {relax}[/dim]")
    if confidence_threshold:
        log(f"[dim]Confidence threshold: {confidence_threshold}[/dim]")
    if model_type == "language":
        log("[dim]Language model: will generate ProtT5 embeddings (slower)[/dim]")
    
    # Determine parallelization strategy early
    use_multiprocess = num_workers > 1 and len(antibodies) >= num_workers * 2
    
    # Load model (only if sequential)
    model = None
    prott5_model = None
    
    if not use_multiprocess:
        start_time = time.time()
        with console.status("[bold green]Loading model..."):
            model = load_model(model_type, device, checkpoint)
            if selected_execution == 'aot':
                from abodybuilder3.aot_runtime import AOTForward
                adapter = AOTForward(aot_root, model_type, model._deployment_checkpoint_sha256, model.forward)
                model.forward = adapter
                model._deployment_stats = adapter.stats
            else:
                try:
                    model._deployment_stats = {'execution': 'eager', 'precision': precision}
                except AttributeError:
                    pass  # Lightweight CPU dispatch fixtures need no mutable model.
        load_time = time.time() - start_time
        log(f"Model loaded in {load_time:.2f}s")
        
        # Load ProtT5 for language model
        if model_type == "language":
            with console.status("[bold green]Loading ProtT5 embedder..."):
                prott5_model = load_prott5(device)
            log("ProtT5 embedder loaded")
    else:
        log("[dim]Skipping main process model load (using worker processes)[/dim]")
    
    # Setup output
    if len(antibodies) > 1:
        output.mkdir(parents=True, exist_ok=True)
    
    # Track used names and results
    name_counts = {}
    used_names = set()
    results = []
    skipped = 0
    failed = 0
    
    def get_unique_name(name: str) -> str:
        count = name_counts.get(name, 0)
        candidate = name if count == 0 else f"{name}_{count}"
        while candidate in used_names:
            count += 1
            candidate = f"{name}_{count}"
        name_counts[name] = count + 1
        used_names.add(candidate)
        return candidate
    
    start_time = time.time()
    
    # Process antibodies
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Predicting structures", total=len(antibodies))
        
        # Parallelization strategy decided earlier

        
        if use_batching:
            run_batched_pipeline(antibodies, model, device, output, output_format,
                                 confidence_threshold, preparation_options, max_batch_size,
                                 batch_memory_gib, cpu_workers, batch_report, get_unique_name,
                                 progress, task)
            return
        elif use_multiprocess:
            # Use ProcessPoolExecutor with each worker having its own model
            from multiprocessing import get_context
            
            log(f"[dim]Using {num_workers} parallel workers for inference[/dim]")
            
            # Prepare work items with unique names
            work_items = []
            for ab_name, heavy_seq, light_seq in antibodies:
                unique_name = get_unique_name(ab_name)
                work_items.append((
                    unique_name, heavy_seq, light_seq,
                    output_format, confidence_threshold
                ))
            
            # Use spawn context to create clean worker processes
            ctx = get_context("spawn")
            checkpoint_path = str(checkpoint) if checkpoint else None
            
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(model_type, checkpoint_path, model_type == "language", precision),
            ) as executor:
                # Submit all work
                futures = {
                    executor.submit(_worker_process_antibody, item): item[0]
                    for item in work_items
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result_name, output_string, mean_plddt = future.result()
                        
                        if output_string is None:
                            skipped += 1
                            progress.update(task, advance=1, description=f"Skipped {result_name}")
                        else:
                            results.append((result_name, output_string, mean_plddt))
                            progress.update(task, advance=1, description=f"Predicted {result_name}")
                    except Exception as e:
                        failed += 1
                        log(f"Failed to predict {name}: {e}", "error")
                        progress.update(task, advance=1)
        else:
            # Sequential processing (small batches or single antibody)
            for ab_name, heavy_seq, light_seq in antibodies:
                try:
                    unique_name = get_unique_name(ab_name)
                    _, output_string, mean_plddt = process_single_antibody(
                        unique_name, heavy_seq, light_seq, model, device,
                        output_format=output_format,
                        confidence_threshold=confidence_threshold,
                        prott5_model=prott5_model,
                    )
                    
                    if output_string is None:
                        skipped += 1
                        progress.update(task, advance=1, description=f"Skipped {unique_name} (pLDDT={mean_plddt:.1f})")
                        continue
                    
                    results.append((unique_name, output_string, mean_plddt))
                    progress.update(task, advance=1, description=f"Predicted {unique_name}")
                except Exception as e:
                    failed += 1
                    log(f"Failed to predict {ab_name}: {e}", "error")
    
    # Write outputs
    ext = {"pdb": ".pdb", "cif": ".cif", "json": ".json"}[output_format.value]
    
    # Determine if output is a directory
    output_is_dir = str(output).endswith('/') or output.is_dir() or len(results) > 1
    
    if output_is_dir:
        # Ensure directory exists
        output.mkdir(parents=True, exist_ok=True)
        
        def write_output(item: Tuple[str, str, Optional[float]]) -> str:
            unique_name, output_string, _ = item
            output_path = output / f"{unique_name}{ext}"
            with open(output_path, "w") as f:
                f.write(output_string)
            if preparation_options is not None:
                write_prepared_outputs(output_string, output_path, preparation_options)
            return unique_name
        
        if len(results) > 1 and preparation_options is None:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(write_output, results))
        else:
            for result in results:
                try:
                    write_output(result)
                except Exception as exc:
                    failed += 1
                    log(f"Preparation failed for {result[0]} (raw prediction retained): {exc}", "error")
    elif results:
        # Single file output
        unique_name, output_string, _ = results[0]
        output_path = output if str(output).endswith(ext) else Path(str(output).rsplit(".", 1)[0] + ext)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_string)
        if preparation_options is not None:
            try:
                write_prepared_outputs(output_string, output_path, preparation_options)
            except Exception as exc:
                failed += 1
                log(f"Preparation failed (raw prediction retained): {exc}", "error")

    elapsed = time.time() - start_time
    rate = len(results) / elapsed if elapsed > 0 else 0
    
    log(f"[green]✓ Done! {len(results)} structures saved to {output}[/green]")
    if skipped:
        log(f"[dim]Skipped {skipped} structures below confidence threshold[/dim]")
    log(f"[dim]Total time: {elapsed:.1f}s ({rate:.1f}/s, {rate*60:.0f}/min)[/dim]")
    if failed:
        log(f"{failed} prediction/preparation failure(s); inspect errors and preparation records.", "error")
        raise typer.Exit(1)


@app.command()
def info():
    """Show model and GPU information."""
    console.print("[bold]ABodyBuilder3 System Info[/bold]\n")
    
    console.print(f"PyTorch version: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    console.print(f"CPU cores: {cpu_count()}")
    console.print(f"Optimal workers: {get_optimal_workers()}")
    
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    console.print("\n[bold]Available Models:[/bold]")
    for model_name in ["plddt", "base", "language"]:
        for base in ["/opt/abodybuilder3", "."]:
            path = Path(base) / f"output/{model_name}-loss/best_second_stage.ckpt"
            if path.exists():
                console.print(f"  ✓ {model_name}: {path}")
                break
        else:
            console.print(f"  ✗ {model_name}: not found")


if __name__ == "__main__":
    app()
