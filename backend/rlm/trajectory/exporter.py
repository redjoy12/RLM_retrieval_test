"""Trajectory Export Module

Provides export functionality for trajectories in multiple formats:
- JSON: Full data export
- HTML: Self-contained report with embedded visualizations
- DOT: GraphViz format for publication-quality diagrams
- PNG/SVG: Image exports (via external rendering if available)
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import base64

import structlog

from rlm.trajectory.processor import (
    TrajectoryProcessor,
    TrajectoryTree,
    TrajectoryNode,
    TrajectoryStepType,
)

logger = structlog.get_logger()


class TrajectoryExporter:
    """Export trajectories in various formats.
    
    Supports:
    - JSON: Complete data dump for debugging/analysis
    - HTML: Self-contained interactive report
    - DOT: GraphViz format for academic papers
    - PNG/SVG: Image exports (requires headless browser if rendering server-side)
    
    Example:
        >>> exporter = TrajectoryExporter(Path("./logs"))
        >>> html = exporter.to_html("session-123")
        >>> dot = exporter.to_dot("session-123")
    """
    
    def __init__(self, log_dir: Path) -> None:
        """Initialize the exporter.
        
        Args:
            log_dir: Directory containing trajectory JSONL files
        """
        self.processor = TrajectoryProcessor(log_dir)
        self.log_dir = Path(log_dir)
        
        logger.info(
            "trajectory_exporter_initialized",
            log_dir=str(log_dir),
        )
    
    def to_json(self, session_id: str, indent: Optional[int] = 2) -> str:
        """Export trajectory as formatted JSON.
        
        Args:
            session_id: Session identifier
            indent: JSON indentation (None for compact)
            
        Returns:
            JSON string with full trajectory data
        """
        tree = self.processor.to_tree(session_id)
        timeline = self.processor.to_timeline(session_id)
        costs = self.processor.get_cost_breakdown(session_id)
        stats = self.processor.get_statistics(session_id)
        
        export_data = {
            "export_metadata": {
                "version": "1.0.0",
                "exported_at": datetime.utcnow().isoformat(),
                "session_id": session_id,
            },
            "tree": tree.to_dict(),
            "timeline": [e.to_dict() for e in timeline],
            "costs": costs.to_dict(),
            "statistics": stats.to_dict(),
        }
        
        return json.dumps(export_data, indent=indent, default=str)
    
    def to_dot(self, session_id: str, rankdir: str = "TB") -> str:
        """Export trajectory as GraphViz DOT format.
        
        Creates a publication-quality hierarchical diagram.
        
        Args:
            session_id: Session identifier
            rankdir: Graph direction (TB=top-bottom, LR=left-right)
            
        Returns:
            DOT format string
        """
        tree = self.processor.to_tree(session_id)
        
        if not tree.nodes:
            return f"// No trajectory data for session {session_id}"
        
        lines = [
            f"digraph Trajectory_{session_id} {{",
            f'    rankdir={rankdir};',
            '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
            '    edge [fontname="Helvetica", fontsize=10];',
            '',
        ]
        
        # Color scheme
        colors = {
            TrajectoryStepType.ROOT_LLM_START: "#3b82f6",      # Blue
            TrajectoryStepType.ROOT_LLM_COMPLETE: "#2563eb",   # Dark Blue
            TrajectoryStepType.CODE_EXECUTION_START: "#eab308", # Yellow
            TrajectoryStepType.CODE_EXECUTION_COMPLETE: "#ca8a04",  # Dark Yellow
            TrajectoryStepType.SUB_LLM_SPAWN: "#22c55e",       # Green
            TrajectoryStepType.SUB_LLM_COMPLETE: "#16a34a",    # Dark Green
            TrajectoryStepType.RECURSION_LIMIT_HIT: "#f97316", # Orange
            TrajectoryStepType.ERROR: "#ef4444",               # Red
            TrajectoryStepType.FINAL_ANSWER: "#6b7280",        # Gray
        }
        
        # Add nodes
        for node_id, node in tree.nodes.items():
            color = colors.get(node.type, "#9ca3af")
            
            # Create label
            label_parts = [
                f"{node.type.value}",
                f"Depth: {node.depth}",
            ]
            
            if node.duration_ms:
                label_parts.append(f"Duration: {node.duration_ms:.0f}ms")
            
            if node.cost.total_tokens > 0:
                label_parts.append(f"Tokens: {node.cost.total_tokens:,}")
                label_parts.append(f"Cost: ${node.cost.cost_usd:.4f}")
            
            label = "\\n".join(label_parts)
            
            # Escape special characters
            label = label.replace('"', '\\"')
            node_id_escaped = node_id.replace('-', '_').replace('.', '_')
            
            lines.append(
                f'    "{node_id_escaped}" [fillcolor="{color}", '
                f'fontcolor="white", label="{label}"];'
            )
        
        lines.append("")
        
        # Add edges
        for node_id, node in tree.nodes.items():
            if node.parent_id:
                parent_escaped = node.parent_id.replace('-', '_').replace('.', '_')
                node_escaped = node_id.replace('-', '_').replace('.', '_')
                lines.append(f'    "{parent_escaped}" -> "{node_escaped}";')
        
        lines.append("}")
        
        return "\\n".join(lines)
    
    def to_html(self, session_id: str, title: Optional[str] = None) -> str:
        """Export trajectory as self-contained HTML report.
        
        Creates a standalone HTML file with embedded visualizations
        that can be viewed in any browser without a server.
        
        Args:
            session_id: Session identifier
            title: Optional custom title
            
        Returns:
            HTML string with embedded visualizations
        """
        tree = self.processor.to_tree(session_id)
        timeline = self.processor.to_timeline(session_id)
        costs = self.processor.get_cost_breakdown(session_id)
        stats = self.processor.get_statistics(session_id)
        
        if not title:
            title = f"RLM Trajectory Report - {session_id}"
        
        # Build tree visualization as HTML/CSS
        tree_html = self._build_tree_html(tree)
        
        # Build timeline as HTML/CSS
        timeline_html = self._build_timeline_html(timeline)
        
        # Build cost summary
        costs_html = self._build_costs_html(costs)
        
        # Build statistics summary
        stats_html = self._build_stats_html(stats)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="meta">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            <p class="meta">Session ID: {session_id}</p>
        </header>
        
        <section class="summary">
            <h2>Execution Summary</h2>
            {stats_html}
        </section>
        
        <section class="costs">
            <h2>Cost Analysis</h2>
            {costs_html}
        </section>
        
        <section class="tree">
            <h2>Execution Tree</h2>
            <div class="tree-container">
                {tree_html}
            </div>
        </section>
        
        <section class="timeline">
            <h2>Timeline</h2>
            <div class="timeline-container">
                {timeline_html}
            </div>
        </section>
        
        <footer>
            <p>Generated by RLM Trajectory Visualizer</p>
        </footer>
    </div>
</body>
</html>'''
        
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML export."""
        return '''
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 
                         'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .meta {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .cost-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .cost-table th,
        .cost-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .cost-table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        
        .tree-node {
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .tree-node.root { border-color: #3b82f6; background: #eff6ff; }
        .tree-node.sub { border-color: #22c55e; background: #f0fdf4; }
        .tree-node.code { border-color: #eab308; background: #fefce8; }
        .tree-node.error { border-color: #ef4444; background: #fef2f2; }
        
        .node-header {
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .node-meta {
            font-size: 0.85em;
            color: #666;
        }
        
        .timeline-item {
            display: flex;
            align-items: center;
            padding: 12px;
            margin: 8px 0;
            background: #f8f9fa;
            border-radius: 6px;
        }
        
        .timeline-time {
            font-family: monospace;
            font-size: 0.85em;
            color: #666;
            min-width: 180px;
        }
        
        .timeline-type {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
            margin: 0 10px;
        }
        
        .timeline-type.root { background: #3b82f6; color: white; }
        .timeline-type.sub { background: #22c55e; color: white; }
        .timeline-type.code { background: #eab308; color: black; }
        .timeline-type.error { background: #ef4444; color: white; }
        
        .timeline-duration {
            margin-left: auto;
            font-size: 0.9em;
            color: #666;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        '''
    
    def _build_tree_html(self, tree: TrajectoryTree) -> str:
        """Build HTML representation of tree."""
        if not tree.nodes:
            return '<p class="empty">No trajectory data available</p>'
        
        def render_node(node_id: str, depth: int = 0) -> str:
            node = tree.nodes.get(node_id)
            if not node:
                return ""
            
            # Determine node class
            if node.depth == 0:
                node_class = "root"
            elif "ERROR" in node.type.value:
                node_class = "error"
            elif "CODE" in node.type.value:
                node_class = "code"
            else:
                node_class = "sub"
            
            # Build node HTML
            html = f'<div class="tree-node {node_class}" style="margin-left: {depth * 20}px;">'
            html += f'<div class="node-header">{node.type.value}</div>'
            
            meta_parts = [f"Depth: {node.depth}"]
            if node.duration_ms:
                meta_parts.append(f"Duration: {node.duration_ms:.0f}ms")
            if node.cost.total_tokens > 0:
                meta_parts.append(f"Tokens: {node.cost.total_tokens:,}")
                meta_parts.append(f"Cost: ${node.cost.cost_usd:.4f}")
            
            html += f'<div class="node-meta">{" | ".join(meta_parts)}</div>'
            html += '</div>'
            
            # Render children
            for child_id in node.children:
                html += render_node(child_id, depth + 1)
            
            return html
        
        return render_node(tree.root_id)
    
    def _build_timeline_html(self, timeline: List[Any]) -> str:
        """Build HTML representation of timeline."""
        if not timeline:
            return '<p class="empty">No timeline data available</p>'
        
        html = ''
        for event in timeline:
            # Determine type class
            if "ROOT" in event.type.value:
                type_class = "root"
            elif "ERROR" in event.type.value:
                type_class = "error"
            elif "CODE" in event.type.value:
                type_class = "code"
            else:
                type_class = "sub"
            
            html += '<div class="timeline-item">'
            html += f'<span class="timeline-time">{event.start_time.strftime("%H:%M:%S.%f")[:-3]}</span>'
            html += f'<span class="timeline-type {type_class}">{event.type.value}</span>'
            
            if event.duration_ms:
                html += f'<span class="timeline-duration">{event.duration_ms:.0f}ms</span>'
            
            html += '</div>'
        
        return html
    
    def _build_costs_html(self, costs: Any) -> str:
        """Build HTML representation of cost breakdown."""
        html = '<div class="cost-summary">'
        html += f'<p><strong>Total Cost:</strong> ${costs.total_cost_usd:.4f}</p>'
        html += f'<p><strong>Total Tokens:</strong> {costs.total_tokens:,} '
        html += f'(Input: {costs.total_input_tokens:,}, Output: {costs.total_output_tokens:,})</p>'
        html += '</div>'
        
        if costs.by_depth:
            html += '<h3>Cost by Depth</h3>'
            html += '<table class="cost-table">'
            html += '<tr><th>Depth</th><th>Count</th><th>Tokens</th><th>Cost</th></tr>'
            
            for depth, data in sorted(costs.by_depth.items()):
                html += f'<tr>'
                html += f'<td>{depth}</td>'
                html += f'<td>{data["count"]}</td>'
                html += f'<td>{data["total_tokens"]:,}</td>'
                html += f'<td>${data["cost_usd"]:.4f}</td>'
                html += f'</tr>'
            
            html += '</table>'
        
        return html
    
    def _build_stats_html(self, stats: Any) -> str:
        """Build HTML representation of statistics."""
        html = '<div class="stats-grid">'
        
        stats_data = [
            ("Total Steps", stats.total_steps),
            ("LLM Calls", stats.total_llm_calls),
            ("Code Executions", stats.total_code_executions),
            ("Max Depth", stats.max_recursion_depth),
            ("Errors", stats.total_errors),
        ]
        
        for label, value in stats_data:
            html += f'<div class="stat-card">'
            html += f'<div class="stat-value">{value}</div>'
            html += f'<div class="stat-label">{label}</div>'
            html += f'</div>'
        
        # Duration
        duration_sec = stats.total_duration_ms / 1000
        html += f'<div class="stat-card">'
        html += f'<div class="stat-value">{duration_sec:.2f}s</div>'
        html += f'<div class="stat-label">Total Duration</div>'
        html += f'</div>'
        
        html += '</div>'
        
        return html
    
    def save_to_file(
        self,
        session_id: str,
        format: str,
        output_path: Path,
    ) -> Path:
        """Export trajectory to a file.
        
        Args:
            session_id: Session identifier
            format: Export format ('json', 'html', 'dot')
            output_path: Path to save file
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        if format == "json":
            content = self.to_json(session_id)
        elif format == "html":
            content = self.to_html(session_id)
        elif format == "dot":
            content = self.to_dot(session_id)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        output_path.write_text(content, encoding="utf-8")
        
        logger.info(
            "trajectory_exported",
            session_id=session_id,
            format=format,
            output_path=str(output_path),
        )
        
        return output_path
