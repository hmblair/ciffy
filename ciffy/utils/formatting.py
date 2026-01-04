"""
Terminal formatting utilities.

Provides ANSI color codes and table formatting functions for terminal output.
"""

from __future__ import annotations
from datetime import date  # noqa: F401 - used in type annotations


# =============================================================================
# ANSI Color Codes
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""

    # Standard colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    GREY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Reset
    RESET = "\033[0m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"


def format_chain_table(
    pdb_id: str | None,
    backend: str,
    rows: list[dict],
    date: "date | None" = None,
) -> str:
    """
    Format chain info as a table string.

    Args:
        pdb_id: PDB identifier, or None if not available.
        backend: Backend name ('numpy' or 'torch').
        rows: List of dicts with keys: 'chain', 'type', 'res', 'atoms'.
        date: Deposition date, or None if not available.

    Returns:
        Formatted table string.
    """
    # Calculate totals first (needed for column width calculation)
    total_res = sum(r['res'] for r in rows)
    total_atoms = sum(r['atoms'] for r in rows)

    # Calculate column widths (include totals in width calculation)
    chain_w = max(1, max((len(r['chain']) for r in rows), default=0))
    type_w = max(4, max((len(r['type']) for r in rows), default=0))
    res_w = max(3, len(str(total_res)), max((len(str(r['res'])) for r in rows), default=0))
    atoms_w = max(5, len(str(total_atoms)), max((len(str(r['atoms'])) for r in rows), default=0))

    # Build header and ensure separator is wide enough for title
    header = f"{'':>{chain_w}}  {'Type':<{type_w}}  {'Res':>{res_w}}  {'Atoms':>{atoms_w}}"
    id_display = pdb_id if pdb_id is not None else ""
    date_display = f" [{date}]" if date is not None else ""
    title = f"Polymer {id_display}{date_display} ({backend})".strip()
    sep_width = max(len(header), len(title))
    sep = "─" * sep_width

    # Build rows
    if pdb_id is not None:
        title_line = f"Polymer {Colors.GREEN}{pdb_id}{Colors.RESET}"
        if date is not None:
            title_line += f" {Colors.GREY}[{date}]{Colors.RESET}"
        title_line += f" {Colors.GREY}({backend}){Colors.RESET}"
    else:
        title_line = f"Polymer {Colors.GREY}({backend}){Colors.RESET}"
    lines = [
        title_line,
        sep,
        header,
        sep,
    ]
    for r in rows:
        res_str = "-" if r['res'] == 0 else str(r['res'])
        lines.append(
            f"{r['chain']:>{chain_w}}  {r['type']:<{type_w}}  {res_str:>{res_w}}  {r['atoms']:>{atoms_w}}"
        )

    # Add totals row only if there are multiple chains
    if len(rows) > 1:
        lines.append(sep)
        num_chains = len(rows)
        lines.append(
            f"{'Σ':>{chain_w}}  {num_chains:<{type_w}}  {total_res:>{res_w}}  {total_atoms:>{atoms_w}}"
        )
    lines.append(sep)

    return "\n".join(lines)


def format_multi_polymer_table(
    polymers: list[tuple[str, "date | None", str, list[dict]]],
    backend: str = "numpy",
) -> str:
    """
    Format chain info from multiple polymers as a unified table with summary.

    Args:
        polymers: List of (pdb_id, date, filepath, rows) tuples where rows is from chain_info().
        backend: Backend name for the header.

    Returns:
        Formatted table string with per-file breakdown and molecule type summary.
    """
    from collections import defaultdict

    if not polymers:
        return "No structures loaded."

    # Collect all rows with their PDB IDs for column width calculation
    all_rows = []
    for pdb_id, date, filepath, rows in polymers:
        for r in rows:
            all_rows.append({**r, 'pdb': pdb_id})

    # Calculate grand totals
    total_res = sum(r['res'] for r in all_rows)
    total_atoms = sum(r['atoms'] for r in all_rows)

    # Calculate column widths (no PDB column - shown as section header instead)
    chain_w = max(5, max((len(r['chain']) for r in all_rows), default=0))
    type_w = max(4, max((len(r['type']) for r in all_rows), default=0))
    res_w = max(3, len(str(total_res)), max((len(str(r['res'])) for r in all_rows), default=0))
    atoms_w = max(5, len(str(total_atoms)), max((len(str(r['atoms'])) for r in all_rows), default=0))

    # Build header
    header = f"{'Chain':<{chain_w}}  {'Type':<{type_w}}  {'Res':>{res_w}}  {'Atoms':>{atoms_w}}"
    sep_width = len(header)
    sep = "─" * sep_width

    lines = []

    # Build rows grouped by PDB
    for idx, (pdb_id, date, filepath, rows) in enumerate(polymers):
        # Add blank line between files (not before first)
        if idx > 0:
            lines.append("")

        # PDB header with date
        date_str = f" {Colors.GREY}[{date}]{Colors.RESET}" if date is not None else ""
        lines.append(f"{Colors.GREEN}{pdb_id}{Colors.RESET}{date_str}")
        lines.append(sep)

        # Calculate per-file totals
        file_res = sum(r['res'] for r in rows)
        file_atoms = sum(r['atoms'] for r in rows)

        for r in rows:
            res_str = "-" if r['res'] == 0 else str(r['res'])
            lines.append(
                f"{r['chain']:<{chain_w}}  {r['type']:<{type_w}}  {res_str:>{res_w}}  {r['atoms']:>{atoms_w}}"
            )

        # Add per-file subtotal if multiple chains
        if len(rows) > 1:
            file_chains = len(rows)
            lines.append(sep)
            lines.append(
                f"{Colors.GREY}{'Σ':<{chain_w}}  {file_chains:<{type_w}}  {file_res:>{res_w}}  {file_atoms:>{atoms_w}}{Colors.RESET}"
            )

    # Grand total (only if multiple files)
    if len(polymers) > 1:
        total_chains = len(all_rows)
        lines.append("")
        lines.append(f"{Colors.BOLD}Total{Colors.RESET}")
        lines.append(sep)
        lines.append(
            f"{Colors.BOLD}{'Σ':<{chain_w}}  {total_chains:<{type_w}}  {total_res:>{res_w}}  {total_atoms:>{atoms_w}}{Colors.RESET}"
        )
        lines.append(sep)

    # Build summary by molecule type
    type_stats = defaultdict(lambda: {'chains': 0, 'res': 0, 'atoms': 0})
    for r in all_rows:
        # Normalize type name (e.g., "MG ION" -> "ION")
        type_name = r['type']
        if ' ION' in type_name:
            type_name = 'ION'
        type_stats[type_name]['chains'] += 1
        type_stats[type_name]['res'] += r['res']
        type_stats[type_name]['atoms'] += r['atoms']

    # Order: RNA, DNA, PROTEIN, then others alphabetically
    priority_order = ['RNA', 'DNA', 'PROTEIN', 'ION', 'LIGAND', 'WATER']
    ordered_types = []
    for t in priority_order:
        if t in type_stats:
            ordered_types.append(t)
    for t in sorted(type_stats.keys()):
        if t not in ordered_types:
            ordered_types.append(t)

    lines.append("")
    lines.append(f"{Colors.BOLD}Summary{Colors.RESET}")

    # Calculate summary column widths
    sum_type_w = max(7, max((len(t) for t in ordered_types), default=0))
    sum_chains_w = max(6, len(str(len(all_rows))))
    sum_res_w = max(3, len(str(total_res)))
    sum_atoms_w = max(5, len(str(total_atoms)))

    sum_header = f"{'Type':<{sum_type_w}}  {'Chains':>{sum_chains_w}}  {'Res':>{sum_res_w}}  {'Atoms':>{sum_atoms_w}}"
    sum_sep = "─" * len(sum_header)

    lines.append(sum_sep)
    lines.append(sum_header)
    lines.append(sum_sep)

    for type_name in ordered_types:
        stats = type_stats[type_name]
        res_str = "-" if stats['res'] == 0 else str(stats['res'])
        lines.append(
            f"{type_name:<{sum_type_w}}  {stats['chains']:>{sum_chains_w}}  {res_str:>{sum_res_w}}  {stats['atoms']:>{sum_atoms_w}}"
        )

    lines.append(sum_sep)
    lines.append(
        f"{Colors.BOLD}{'Σ':<{sum_type_w}}  {len(all_rows):>{sum_chains_w}}  {total_res:>{sum_res_w}}  {total_atoms:>{sum_atoms_w}}{Colors.RESET}"
    )
    lines.append(sum_sep)

    return "\n".join(lines)
