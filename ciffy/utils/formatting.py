"""
Terminal formatting utilities.

Provides ANSI color codes and table formatting functions for terminal output.
"""


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


def format_chain_table(pdb_id: str, backend: str, rows: list[dict]) -> str:
    """
    Format chain info as a table string.

    Args:
        pdb_id: PDB identifier.
        backend: Backend name ('numpy' or 'torch').
        rows: List of dicts with keys: 'chain', 'type', 'res', 'atoms'.

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
    title = f"Polymer {pdb_id} ({backend})"
    sep_width = max(len(header), len(title))
    sep = "─" * sep_width

    # Build rows
    lines = [
        f"Polymer {Colors.GREEN}{pdb_id}{Colors.RESET} {Colors.GREY}({backend}){Colors.RESET}",
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
        lines.append(
            f"{'Σ':>{chain_w}}  {'':<{type_w}}  {total_res:>{res_w}}  {total_atoms:>{atoms_w}}"
        )
    lines.append(sep)

    return "\n".join(lines)
