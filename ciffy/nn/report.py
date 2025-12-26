"""
Training report generation.

Generates HTML or Markdown reports summarizing training results.
Designed for molecular biology researchers who need to document
and share their model training.

Example:
    >>> from ciffy.nn.report import TrainingReport
    >>>
    >>> report = TrainingReport(
    ...     model_type="flow",
    ...     config=config,
    ...     results=results,
    ... )
    >>> report.save("training_report.html")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TrainingReport:
    """Training report with results and recommendations.

    Generates HTML or Markdown reports summarizing:
    - Training configuration
    - Per-residue/model results
    - Metrics and visualizations
    - Recommendations for next steps
    """

    model_type: str
    config: dict[str, Any]
    results: dict[str, Any]
    title: str | None = None
    notes: str | None = None

    # Timing info
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # Loss history (for plotting)
    loss_history: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.title is None:
            self.title = f"{self.model_type.title()} Training Report"

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins} min {secs} sec"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours} hr {mins} min"

    def _generate_html(self) -> str:
        """Generate HTML report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build config table
        config_rows = []
        for section, values in self.config.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    config_rows.append(f"<tr><td>{section}.{key}</td><td>{val}</td></tr>")
            else:
                config_rows.append(f"<tr><td>{section}</td><td>{values}</td></tr>")

        # Build results table
        results_rows = []
        if self.model_type == "flow":
            residue_results = self.results.get("extra_metrics", {}).get("residue_results", {})
            for residue, metrics in residue_results.items():
                results_rows.append(f"""
                <tr>
                    <td><strong>{residue}</strong></td>
                    <td>{metrics.get('n_train', 0):,}</td>
                    <td>{metrics.get('n_test', 0):,}</td>
                    <td>{metrics.get('pca_rmsd', 0):.4f}Å</td>
                    <td>{metrics.get('test_rmsd', 0):.4f}Å</td>
                    <td>{metrics.get('var_explained', 0)*100:.1f}%</td>
                </tr>
                """)

        # Generate loss chart if we have data
        loss_chart_html = ""
        if self.loss_history:
            loss_chart_html = self._generate_loss_chart_html()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }}
        h2 {{ color: #34495e; margin-top: 2rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .metric {{ font-family: monospace; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .info {{ color: #3498db; }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9rem; }}
        ul {{ padding-left: 1.5rem; }}
        li {{ margin: 0.5rem 0; }}
        .chart {{ margin: 1rem 0; background: white; padding: 1rem; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p class="timestamp">Generated: {timestamp}</p>

    <div class="card">
        <strong>Status:</strong> <span class="success">✓ {self.results.get('status', 'completed').title()}</span><br>
        <strong>Duration:</strong> {self._format_duration(self.duration_seconds)}<br>
        <strong>Model type:</strong> {self.model_type}<br>
        <strong>Epochs:</strong> {self.results.get('epochs_trained', 0)}/{self.results.get('total_epochs', 0)}
    </div>

    <h2>Results</h2>
    <table>
        <thead>
            <tr>
                <th>Residue</th>
                <th>Train</th>
                <th>Test</th>
                <th>PCA RMSD</th>
                <th>Test RMSD</th>
                <th>Var Explained</th>
            </tr>
        </thead>
        <tbody>
            {''.join(results_rows) if results_rows else '<tr><td colspan="6">No per-residue results available</td></tr>'}
        </tbody>
    </table>

    {loss_chart_html}

    <h2>Configuration</h2>
    <table>
        <thead>
            <tr><th>Parameter</th><th>Value</th></tr>
        </thead>
        <tbody>
            {''.join(config_rows)}
        </tbody>
    </table>

    {f'<h2>Notes</h2><p>{self.notes}</p>' if self.notes else ''}

    <hr>
    <p class="timestamp">
        Output directory: {self.results.get('checkpoint_path', 'N/A')}<br>
        Generated by ciffy training report
    </p>
</body>
</html>"""
        return html

    def _generate_loss_chart_html(self) -> str:
        """Generate inline SVG loss chart."""
        if not self.loss_history:
            return ""

        # Simple SVG chart
        width, height = 600, 200
        padding = 40

        charts = []
        for name, losses in self.loss_history.items():
            if not losses:
                continue

            n_points = len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            loss_range = max_loss - min_loss or 1

            # Generate path
            points = []
            for i, loss in enumerate(losses):
                x = padding + (i / max(n_points - 1, 1)) * (width - 2 * padding)
                y = height - padding - ((loss - min_loss) / loss_range) * (height - 2 * padding)
                points.append(f"{x:.1f},{y:.1f}")

            path = "M " + " L ".join(points)

            charts.append(f"""
            <div class="chart">
                <strong>{name} Loss</strong>
                <svg width="{width}" height="{height}" style="display: block;">
                    <!-- Axes -->
                    <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#ccc"/>
                    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#ccc"/>

                    <!-- Loss curve -->
                    <path d="{path}" fill="none" stroke="#3498db" stroke-width="2"/>

                    <!-- Labels -->
                    <text x="{width//2}" y="{height-5}" text-anchor="middle" font-size="12" fill="#666">Epoch</text>
                    <text x="10" y="{height//2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90 10 {height//2})">Loss</text>
                    <text x="{padding}" y="{height-padding+15}" font-size="10" fill="#666">0</text>
                    <text x="{width-padding}" y="{height-padding+15}" font-size="10" fill="#666">{n_points}</text>
                    <text x="{padding-5}" y="{padding}" font-size="10" fill="#666" text-anchor="end">{max_loss:.2f}</text>
                    <text x="{padding-5}" y="{height-padding}" font-size="10" fill="#666" text-anchor="end">{min_loss:.2f}</text>
                </svg>
            </div>
            """)

        return "\n".join(charts)

    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# {self.title}",
            "",
            f"*Generated: {timestamp}*",
            "",
            "## Summary",
            "",
            f"- **Status:** {self.results.get('status', 'completed').title()}",
            f"- **Duration:** {self._format_duration(self.duration_seconds)}",
            f"- **Model type:** {self.model_type}",
            f"- **Epochs:** {self.results.get('epochs_trained', 0)}/{self.results.get('total_epochs', 0)}",
            "",
        ]

        # Results table
        if self.model_type == "flow":
            residue_results = self.results.get("extra_metrics", {}).get("residue_results", {})
            if residue_results:
                lines.extend([
                    "## Results",
                    "",
                    "| Residue | Train | Test | PCA RMSD | Test RMSD | Var% |",
                    "|---------|-------|------|----------|-----------|------|",
                ])
                for residue, metrics in residue_results.items():
                    lines.append(
                        f"| {residue} | {metrics.get('n_train', 0):,} | "
                        f"{metrics.get('n_test', 0):,} | "
                        f"{metrics.get('pca_rmsd', 0):.4f}Å | "
                        f"{metrics.get('test_rmsd', 0):.4f}Å | "
                        f"{metrics.get('var_explained', 0)*100:.1f}% |"
                    )
                lines.append("")

        # Configuration
        lines.extend([
            "## Configuration",
            "",
            "```yaml",
        ])
        for section, values in self.config.items():
            if isinstance(values, dict):
                lines.append(f"{section}:")
                for key, val in values.items():
                    lines.append(f"  {key}: {val}")
            else:
                lines.append(f"{section}: {values}")
        lines.extend([
            "```",
            "",
        ])

        # Notes
        if self.notes:
            lines.extend([
                "## Notes",
                "",
                self.notes,
                "",
            ])

        # Footer
        lines.extend([
            "---",
            "",
            f"Output directory: `{self.results.get('checkpoint_path', 'N/A')}`",
        ])

        return "\n".join(lines)

    def save(self, path: str | Path, format: str | None = None) -> Path:
        """Save report to file.

        Args:
            path: Output file path.
            format: 'html' or 'md'. If None, inferred from extension.

        Returns:
            Path to saved file.
        """
        path = Path(path)

        # Infer format from extension
        if format is None:
            if path.suffix.lower() in (".html", ".htm"):
                format = "html"
            elif path.suffix.lower() in (".md", ".markdown"):
                format = "md"
            else:
                format = "html"
                path = path.with_suffix(".html")

        # Generate content
        if format == "html":
            content = self._generate_html()
        else:
            content = self._generate_markdown()

        # Write file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        return path

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "model_type": self.model_type,
            "config": self.config,
            "results": self.results,
            "duration_seconds": self.duration_seconds,
            "generated_at": datetime.now().isoformat(),
        }

    def save_json(self, path: str | Path) -> Path:
        """Save report as JSON.

        Args:
            path: Output file path.

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def generate_flow_report(
    config: dict[str, Any],
    results: dict[str, Any],
    output_path: str | Path,
    duration_seconds: float = 0.0,
    loss_history: dict[str, list[float]] | None = None,
) -> Path:
    """Convenience function to generate a flow training report.

    Args:
        config: Training configuration dict.
        results: Training results dict.
        output_path: Where to save the report.
        duration_seconds: Total training duration.
        loss_history: Optional dict of loss histories by residue.

    Returns:
        Path to saved report.
    """
    report = TrainingReport(
        model_type="flow",
        config=config,
        results=results,
        duration_seconds=duration_seconds,
        loss_history=loss_history or {},
    )
    return report.save(output_path)


__all__ = [
    "TrainingReport",
    "generate_flow_report",
]
