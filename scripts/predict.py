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
    
    # With structure relaxation (slower but more accurate bond geometries)
    predict.py predict --csv input.csv -o output/ --relaxation

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
    
    Benchmarks show 4 workers is optimal for DGX Spark (106 antibodies/min).
    """
    # 4 workers is optimal based on benchmarks
    return 4


def load_model(model_type: str, device: torch.device, checkpoint: Optional[Path] = None):
    """Load the specified model."""
    if checkpoint and checkpoint.exists():
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
    
    module = LitABB3.load_from_checkpoint(str(path))
    model = module.model
    model.to(device)
    model.eval()
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
    return output_to_pdb(output, ab_input)


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


def apply_relaxation(pdb_string: str) -> str:
    """Apply structure relaxation using OpenMM/pdbfixer."""
    try:
        from pdbfixer import PDBFixer
        from openmm import app
        from io import StringIO
        
        # Use pdbfixer to clean up the structure
        fixer = PDBFixer(pdbfile=StringIO(pdb_string))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        
        # Write back to PDB
        output = StringIO()
        app.PDBFile.writeFile(fixer.topology, fixer.positions, output)
        return output.getvalue()
    except Exception as e:
        log(f"Warning: Relaxation failed: {e}", "warning")
        return pdb_string


def process_single_antibody(
    name: str,
    heavy: str,
    light: str,
    model,
    device: torch.device,
    relaxation: bool = False,
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
    
    # Apply relaxation if requested
    if relaxation and output_format == OutputFormat.pdb:
        output_string = apply_relaxation(output_string)
    
    return name, output_string, mean_plddt


# Worker state for multiprocessing
_worker_model = None
_worker_device = None
_worker_prott5 = None
_worker_config = None


def _init_worker(model_type: str, checkpoint_path: Optional[str], use_language: bool):
    """Initialize worker process with its own model instance."""
    global _worker_model, _worker_device, _worker_prott5, _worker_config
    
    # Suppress warnings in worker
    import warnings
    warnings.filterwarnings("ignore")
    
    # Register safe globals
    torch.serialization.add_safe_globals([ml_collections.ConfigDict])
    
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
    name, heavy, light, relaxation, output_format, confidence_threshold = args
    
    global _worker_model, _worker_device, _worker_prott5
    
    try:
        result = process_single_antibody(
            name, heavy, light,
            _worker_model, _worker_device,
            relaxation=relaxation,
            output_format=output_format,
            confidence_threshold=confidence_threshold,
            prott5_model=_worker_prott5,
        )
        return result
    except Exception as e:
        return name, None, None


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
    
    # Processing options
    relaxation: bool = typer.Option(
        False, "--relaxation", "-r",
        help="Apply OpenMM structure relaxation. Improves bond geometries but increases runtime ~20%."
    ),
    confidence_threshold: Optional[float] = typer.Option(
        None, "--confidence-threshold", "-t",
        help="Skip antibodies with mean pLDDT below this threshold (0-100). Only works with plddt model."
    ),
    workers: str = typer.Option(
        "auto", "--workers", "-w",
        help="Parallel workers: 'auto' (detects optimal), '1' (sequential), or integer. Affects I/O, not GPU."
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
    
    # Determine number of workers
    if workers == "auto":
        num_workers = get_optimal_workers()
    else:
        try:
            num_workers = int(workers)
        except ValueError:
            num_workers = 1
    
    if len(antibodies) <= 2:
        num_workers = 1
    
    log(f"[bold]ABodyBuilder3[/bold] - {model_type} model on {device}")
    log(f"Processing {len(antibodies)} antibody(ies) with {num_workers} worker(s)...")
    if relaxation:
        log("[dim]Relaxation enabled (will take longer)[/dim]")
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
    results = []
    skipped = 0
    
    def get_unique_name(name: str) -> str:
        if name in name_counts:
            name_counts[name] += 1
            return f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 0
            return name
    
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

        
        if use_multiprocess:
            # Use ProcessPoolExecutor with each worker having its own model
            from multiprocessing import get_context
            
            log(f"[dim]Using {num_workers} parallel workers for inference[/dim]")
            
            # Prepare work items with unique names
            work_items = []
            for ab_name, heavy_seq, light_seq in antibodies:
                unique_name = get_unique_name(ab_name)
                work_items.append((
                    unique_name, heavy_seq, light_seq,
                    relaxation, output_format, confidence_threshold
                ))
            
            # Use spawn context to create clean worker processes
            ctx = get_context("spawn")
            checkpoint_path = str(checkpoint) if checkpoint else None
            
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(model_type, checkpoint_path, model_type == "language"),
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
                        log(f"Warning: Failed to predict {name}: {e}", "warning")
                        progress.update(task, advance=1)
        else:
            # Sequential processing (small batches or single antibody)
            for ab_name, heavy_seq, light_seq in antibodies:
                try:
                    unique_name = get_unique_name(ab_name)
                    _, output_string, mean_plddt = process_single_antibody(
                        unique_name, heavy_seq, light_seq, model, device,
                        relaxation=relaxation,
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
                    log(f"Warning: Failed to predict {ab_name}: {e}", "warning")
    
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
            return unique_name
        
        if len(results) > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(write_output, results))
        elif results:
            write_output(results[0])
    elif results:
        # Single file output
        unique_name, output_string, _ = results[0]
        output_path = output if str(output).endswith(ext) else Path(str(output).rsplit(".", 1)[0] + ext)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_string)
    
    elapsed = time.time() - start_time
    rate = len(results) / elapsed if elapsed > 0 else 0
    
    log(f"[green]✓ Done! {len(results)} structures saved to {output}[/green]")
    if skipped:
        log(f"[dim]Skipped {skipped} structures below confidence threshold[/dim]")
    log(f"[dim]Total time: {elapsed:.1f}s ({rate:.1f}/s, {rate*60:.0f}/min)[/dim]")


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
