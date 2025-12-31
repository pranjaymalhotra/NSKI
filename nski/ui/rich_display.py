#!/usr/bin/env python3
"""
Rich Terminal Display for NSKI Pipeline

Beautiful, informative terminal output with:
- Progress bars with ETA
- Live stats dashboard
- GPU/Memory monitoring
- LLM commentary
- Publication-ready logging

Author: Pranjay Malhotra
Date: December 2024
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NSKIDashboard:
    """Rich terminal dashboard for NSKI experiments."""
    
    def __init__(self, output_dir: Path = None):
        self.console = Console() if HAS_RICH else None
        self.output_dir = output_dir
        self.start_time = time.time()
        self.current_model = ""
        self.current_method = ""
        self.current_step = 0
        self.total_steps = 12
        self.results: List[Dict] = []
        self.llm_comments: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Stats
        self.models_completed = 0
        self.total_models = 0
        self.prompts_processed = 0
        self.total_prompts = 0
        self.best_asr = 1.0
        self.best_method = ""
        
    def print_banner(self):
        """Print beautiful startup banner."""
        if not HAS_RICH:
            print("=" * 70)
            print("NSKI: Neural Surgical Key-Value Intervention")
            print("Publication-Ready Research Pipeline")
            print("=" * 70)
            return
            
        banner = """
[bold blue]╔══════════════════════════════════════════════════════════════════════════════╗[/]
[bold blue]║[/]                                                                              [bold blue]║[/]
[bold blue]║[/]   [bold white]███╗   ██╗███████╗██╗  ██╗██╗[/]   [bold green]Neural Surgical Key-Value Intervention[/]  [bold blue]║[/]
[bold blue]║[/]   [bold white]████╗  ██║██╔════╝██║ ██╔╝██║[/]   [dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]  [bold blue]║[/]
[bold blue]║[/]   [bold white]██╔██╗ ██║███████╗█████╔╝ ██║[/]   [yellow]Publication-Ready Research Pipeline[/]    [bold blue]║[/]
[bold blue]║[/]   [bold white]██║╚██╗██║╚════██║██╔═██╗ ██║[/]   [dim]Real KV-Cache Interventions[/]             [bold blue]║[/]
[bold blue]║[/]   [bold white]██║ ╚████║███████║██║  ██╗██║[/]   [dim]Statistical Validation[/]                  [bold blue]║[/]
[bold blue]║[/]   [bold white]╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝[/]   [dim]Multi-Model Benchmarking[/]                [bold blue]║[/]
[bold blue]║[/]                                                                              [bold blue]║[/]
[bold blue]╚══════════════════════════════════════════════════════════════════════════════╝[/]
"""
        self.console.print(banner)
        
    def print_config(self, config: Dict):
        """Print configuration in a nice table."""
        if not HAS_RICH:
            print("\nConfiguration:")
            for k, v in config.items():
                print(f"  {k}: {v}")
            return
            
        table = Table(title="[bold]Experiment Configuration[/]", box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value[:3])
                if len(config.get(key, [])) > 3:
                    value += f" (+{len(config.get(key, [])) - 3} more)"
            table.add_row(str(key), str(value))
        
        self.console.print(table)
        
    def get_gpu_stats(self) -> Dict:
        """Get current GPU statistics."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_used": torch.cuda.memory_allocated(0) / 1e9,
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "memory_percent": torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100,
        }
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
    
    def get_eta(self) -> str:
        """Calculate estimated time remaining."""
        elapsed = time.time() - self.start_time
        
        if self.models_completed == 0:
            return "Calculating..."
        
        time_per_model = elapsed / self.models_completed
        remaining_models = self.total_models - self.models_completed
        eta_seconds = remaining_models * time_per_model
        
        return self.format_time(eta_seconds)
    
    def create_stats_panel(self) -> Panel:
        """Create statistics panel."""
        gpu = self.get_gpu_stats()
        elapsed = time.time() - self.start_time
        
        stats = Table.grid(padding=1)
        stats.add_column(justify="right", style="cyan")
        stats.add_column(justify="left")
        
        stats.add_row("Elapsed:", self.format_time(elapsed))
        stats.add_row("ETA:", self.get_eta())
        stats.add_row("Models:", f"{self.models_completed}/{self.total_models}")
        stats.add_row("Prompts:", f"{self.prompts_processed:,}/{self.total_prompts:,}")
        
        if gpu["available"]:
            stats.add_row("GPU:", f"{gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB ({gpu['memory_percent']:.0f}%)")
        
        if self.best_asr < 1.0:
            stats.add_row("Best ASR:", f"{self.best_asr:.3f} ({self.best_method})")
        
        return Panel(stats, title="[bold]Statistics[/]", border_style="blue")
    
    def create_results_table(self) -> Table:
        """Create results table."""
        table = Table(title="[bold]Results So Far[/]", box=box.SIMPLE)
        table.add_column("Model", style="cyan", max_width=25)
        table.add_column("Method", style="yellow")
        table.add_column("ASR ↓", justify="right", style="red")
        table.add_column("Refusal ↑", justify="right", style="green")
        table.add_column("PPL", justify="right")
        
        for r in self.results[-10:]:  # Last 10 results
            model_short = r.get("model", "").split("/")[-1][:20]
            asr = r.get("asr", 0)
            refusal = r.get("refusal_rate", 0)
            ppl = r.get("perplexity", 0)
            
            asr_style = "bold green" if asr < 0.1 else ("yellow" if asr < 0.3 else "red")
            
            table.add_row(
                model_short,
                r.get("method", "?"),
                f"[{asr_style}]{asr:.3f}[/]",
                f"{refusal:.3f}",
                f"{ppl:.1f}" if ppl else "N/A"
            )
        
        return table
    
    def step(self, step_num: int, total: int, description: str):
        """Log a pipeline step."""
        self.current_step = step_num
        self.total_steps = total
        
        if not HAS_RICH:
            print(f"\n[STEP {step_num}/{total}] {description}")
            print("=" * 60)
            return
        
        self.console.print()
        self.console.rule(f"[bold blue]Step {step_num}/{total}: {description}[/]")
    
    def progress(self, message: str):
        """Log progress message."""
        if not HAS_RICH:
            print(f"  → {message}")
            return
        self.console.print(f"  [dim]→[/] {message}")
    
    def success(self, message: str):
        """Log success message."""
        if not HAS_RICH:
            print(f"  ✓ {message}")
            return
        self.console.print(f"  [green]✓[/] {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        self.warnings.append(message)
        if not HAS_RICH:
            print(f"  ⚠ {message}")
            return
        self.console.print(f"  [yellow]⚠[/] {message}")
    
    def error(self, message: str):
        """Log error message."""
        self.errors.append(message)
        if not HAS_RICH:
            print(f"  ✗ {message}")
            return
        self.console.print(f"  [red]✗[/] {message}")
    
    def add_result(self, result: Dict):
        """Add experiment result."""
        self.results.append(result)
        asr = result.get("asr", 1.0)
        if asr < self.best_asr:
            self.best_asr = asr
            self.best_method = result.get("method", "unknown")
    
    def add_llm_comment(self, comment: str):
        """Add LLM commentary."""
        self.llm_comments.append(comment)
        if not HAS_RICH:
            print(f"\n  🤖 LLM: {comment}\n")
            return
        self.console.print(Panel(
            Markdown(comment),
            title="[bold magenta]🤖 LLM Analysis[/]",
            border_style="magenta"
        ))
    
    def show_interim_summary(self):
        """Show interim summary during long runs."""
        if not HAS_RICH:
            print("\n" + "=" * 60)
            print("INTERIM SUMMARY")
            print("=" * 60)
            print(f"Results so far: {len(self.results)}")
            print(f"Best ASR: {self.best_asr:.3f} ({self.best_method})")
            return
        
        self.console.print()
        self.console.print(self.create_stats_panel())
        if self.results:
            self.console.print(self.create_results_table())
    
    def create_progress_bar(self, description: str, total: int) -> Progress:
        """Create a progress bar for iterations."""
        if not HAS_RICH:
            return None
        
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
    
    def print_final_summary(self, all_results: Dict):
        """Print final comprehensive summary."""
        if not HAS_RICH:
            print("\n" + "=" * 70)
            print("FINAL RESULTS SUMMARY")
            print("=" * 70)
            return
        
        elapsed = time.time() - self.start_time
        
        # Header
        self.console.print()
        self.console.rule("[bold green]EXPERIMENT COMPLETE[/]")
        
        # Summary stats
        summary = Table.grid(padding=1)
        summary.add_column(justify="right", style="cyan")
        summary.add_column(justify="left", style="white")
        
        summary.add_row("Total Time:", self.format_time(elapsed))
        summary.add_row("Models Tested:", str(self.models_completed))
        summary.add_row("Total Experiments:", str(len(self.results)))
        summary.add_row("Best ASR:", f"{self.best_asr:.3f} ({self.best_method})")
        summary.add_row("Errors:", str(len(self.errors)))
        summary.add_row("Warnings:", str(len(self.warnings)))
        
        self.console.print(Panel(summary, title="[bold]Summary[/]", border_style="green"))
        
        # Final results table
        if self.results:
            self.console.print(self.create_results_table())


def install_rich_if_needed():
    """Install rich library if not available."""
    global HAS_RICH
    if HAS_RICH:
        return True
    
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
        
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress
        from rich.table import Table
        from rich.layout import Layout
        from rich.live import Live
        from rich.text import Text
        from rich import box
        
        HAS_RICH = True
        return True
    except Exception:
        return False
