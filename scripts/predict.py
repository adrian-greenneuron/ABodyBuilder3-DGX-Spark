#!/usr/bin/env python3
"""
ABodyBuilder3 Command Line Interface

Predict antibody structures from sequences via command line.

Usage:
    # Single antibody (sequences on command line)
    python predict.py predict --heavy "QVQL..." --light "DIQM..." -o output.pdb

    # From FASTA file (paired heavy/light chains)
    python predict.py predict --fasta antibodies.fasta -o output_dir/

    # CSV input with multiple antibodies (with multiprocessing)
    python predict.py predict --csv antibodies.csv -o output_dir/ --workers auto
    
    # With relaxation for higher accuracy
    python predict.py predict --csv antibodies.csv -o output_dir/ --relaxation
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    pdb = "pdb"
    cif = "cif"
    json = "json"


app = typer.Typer(
    help="ABodyBuilder3: Predict antibody 3D structures from sequence",
    add_completion=False,
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
    """Determine optimal number of workers based on system specs."""
    num_cpus = cpu_count()
    
    # Get available memory (rough heuristic)
    try:
        import psutil
        mem_gb = psutil.virtual_memory().available / 1e9
        mem_workers = int(mem_gb / 0.5)
    except ImportError:
        mem_workers = num_cpus
    
    # Leave 2 cores for system/GPU operations
    optimal = min(num_cpus - 2, mem_workers)
    return max(1, optimal)


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
) -> Tuple[str, str, Optional[float]]:
    """Process a single antibody end-to-end. Returns (name, output_string, mean_plddt)."""
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
    heavy: Optional[str] = typer.Option(None, "--heavy", "-H", help="Heavy chain sequence"),
    light: Optional[str] = typer.Option(None, "--light", "-L", help="Light chain sequence"),
    name: str = typer.Option("antibody", "--name", "-n", help="Antibody name (for --heavy/--light input)"),
    fasta: Optional[Path] = typer.Option(None, "--fasta", "-f", help="FASTA file with paired heavy/light chains"),
    csv_file: Optional[Path] = typer.Option(None, "--csv", "-c", help="CSV file with name,heavy,light columns"),
    
    # Output options
    output: Path = typer.Option("output.pdb", "--output", "-o", help="Output PDB file or directory"),
    output_format: OutputFormat = typer.Option(OutputFormat.pdb, "--format", help="Output format: pdb, cif, or json"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output files"),
    
    # Model options
    model_type: str = typer.Option("plddt", "--model", "-m", help="Model type: plddt, base, or language"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="Custom checkpoint path"),
    device_str: str = typer.Option("cuda", "--device", "-d", help="Device: cuda or cpu"),
    
    # Processing options
    relaxation: bool = typer.Option(False, "--relaxation", "-r", help="Apply structure relaxation (slower but more accurate)"),
    confidence_threshold: Optional[float] = typer.Option(None, "--confidence-threshold", "-t", help="Filter outputs by minimum pLDDT score (0-100)"),
    workers: str = typer.Option("auto", "--workers", "-w", help="Number of workers (auto, 1, or integer)"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility"),
    
    # Output verbosity
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Predict antibody structure from sequence."""
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
        if output.exists() and not overwrite:
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
    
    # Load model
    start_time = time.time()
    with console.status("[bold green]Loading model..."):
        model = load_model(model_type, device, checkpoint)
    load_time = time.time() - start_time
    log(f"Model loaded in {load_time:.2f}s")
    
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
        
        if num_workers > 1 and len(antibodies) > 2:
            # Parallel input preparation
            prepared_inputs = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(prepare_input, heavy_seq, light_seq): (ab_name, heavy_seq, light_seq)
                    for ab_name, heavy_seq, light_seq in antibodies
                }
                
                for future in as_completed(futures):
                    ab_name, heavy_seq, light_seq = futures[future]
                    try:
                        ab_input = future.result()
                        prepared_inputs.append((ab_name, heavy_seq, light_seq, ab_input))
                    except Exception as e:
                        log(f"Warning: Failed to prepare {ab_name}: {e}", "warning")
            
            # Sequential GPU inference
            for ab_name, heavy_seq, light_seq, ab_input in prepared_inputs:
                try:
                    unique_name = get_unique_name(ab_name)
                    output_dict = run_inference(model, ab_input, device)
                    
                    plddt_scores = extract_plddt_scores(output_dict)
                    mean_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else None
                    
                    if confidence_threshold and mean_plddt and mean_plddt < confidence_threshold:
                        skipped += 1
                        progress.update(task, advance=1, description=f"Skipped {unique_name} (pLDDT={mean_plddt:.1f})")
                        continue
                    
                    if output_format == OutputFormat.json:
                        output_string = json.dumps(generate_json_output(unique_name, heavy_seq, light_seq, output_dict, ab_input), indent=2)
                    elif output_format == OutputFormat.cif:
                        output_string = generate_cif(output_dict, ab_input, unique_name)
                    else:
                        output_string = generate_pdb(output_dict, ab_input)
                    
                    if relaxation and output_format == OutputFormat.pdb:
                        output_string = apply_relaxation(output_string)
                    
                    results.append((unique_name, output_string, mean_plddt))
                    progress.update(task, advance=1, description=f"Predicted {unique_name}")
                except Exception as e:
                    log(f"Warning: Failed to predict {ab_name}: {e}", "warning")
        else:
            # Sequential processing
            for ab_name, heavy_seq, light_seq in antibodies:
                try:
                    unique_name = get_unique_name(ab_name)
                    _, output_string, mean_plddt = process_single_antibody(
                        unique_name, heavy_seq, light_seq, model, device,
                        relaxation=relaxation,
                        output_format=output_format,
                        confidence_threshold=confidence_threshold,
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
    
    if len(results) > 1:
        def write_output(item: Tuple[str, str, Optional[float]]) -> str:
            unique_name, output_string, _ = item
            output_path = output / f"{unique_name}{ext}"
            with open(output_path, "w") as f:
                f.write(output_string)
            return unique_name
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(write_output, results))
    elif results:
        unique_name, output_string, _ = results[0]
        output_path = output if str(output).endswith(ext) else Path(str(output).rsplit(".", 1)[0] + ext)
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
