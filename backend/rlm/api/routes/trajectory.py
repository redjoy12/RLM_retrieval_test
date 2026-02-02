"""Trajectory API Routes

REST API endpoints for trajectory visualization and export.
Provides access to trajectory data in various formats.
"""

from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse

import structlog

from rlm.config import get_settings
from rlm.trajectory.processor import TrajectoryProcessor
from rlm.trajectory.exporter import TrajectoryExporter

logger = structlog.get_logger()

router = APIRouter(prefix="/trajectory", tags=["trajectory"])

# Initialize processor and exporter
settings = get_settings()
processor = TrajectoryProcessor(Path(settings.log_dir))
exporter = TrajectoryExporter(Path(settings.log_dir))


@router.get("/{session_id}")
async def get_trajectory(session_id: str) -> JSONResponse:
    """Get full trajectory data.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Complete trajectory data including tree, timeline, costs, and statistics
    """
    logger.info("get_trajectory_request", session_id=session_id)
    
    try:
        tree = processor.to_tree(session_id)
        timeline = processor.to_timeline(session_id)
        costs = processor.get_cost_breakdown(session_id)
        stats = processor.get_statistics(session_id)
        
        return JSONResponse(content={
            "session_id": session_id,
            "tree": tree.to_dict(),
            "timeline": [e.to_dict() for e in timeline],
            "costs": costs.to_dict(),
            "statistics": stats.to_dict(),
        })
    
    except Exception as e:
        logger.error(
            "get_trajectory_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/tree")
async def get_trajectory_tree(session_id: str) -> JSONResponse:
    """Get trajectory as tree structure.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Tree structure optimized for React Flow visualization
    """
    logger.info("get_trajectory_tree_request", session_id=session_id)
    
    try:
        tree = processor.to_tree(session_id)
        
        # Convert to React Flow format
        nodes = []
        edges = []
        
        for node_id, node in tree.nodes.items():
            # Determine node style based on type
            node_styles = {
                "ROOT_LLM_START": {"background": "#3b82f6", "color": "white"},
                "ROOT_LLM_COMPLETE": {"background": "#2563eb", "color": "white"},
                "CODE_EXECUTION_START": {"background": "#eab308", "color": "black"},
                "CODE_EXECUTION_COMPLETE": {"background": "#ca8a04", "color": "white"},
                "SUB_LLM_SPAWN": {"background": "#22c55e", "color": "white"},
                "SUB_LLM_COMPLETE": {"background": "#16a34a", "color": "white"},
                "RECURSION_LIMIT_HIT": {"background": "#f97316", "color": "white"},
                "ERROR": {"background": "#ef4444", "color": "white"},
                "FINAL_ANSWER": {"background": "#6b7280", "color": "white"},
            }
            
            style = node_styles.get(node.type.value, {"background": "#9ca3af", "color": "white"})
            
            # Build label
            label_lines = [node.type.value]
            if node.duration_ms:
                label_lines.append(f"⏱️ {node.duration_ms:.0f}ms")
            if node.cost.total_tokens > 0:
                label_lines.append(f"🪙 {node.cost.total_tokens:,} tokens")
                label_lines.append(f"💰 ${node.cost.cost_usd:.4f}")
            
            nodes.append({
                "id": node_id,
                "type": "trajectoryNode",
                "position": {"x": node.depth * 250, "y": 0},  # Will be laid out by React Flow
                "data": {
                    "label": "\\n".join(label_lines),
                    "node_type": node.type.value,
                    "depth": node.depth,
                    "duration_ms": node.duration_ms,
                    "cost": {
                        "input_tokens": node.cost.input_tokens,
                        "output_tokens": node.cost.output_tokens,
                        "total_tokens": node.cost.total_tokens,
                        "cost_usd": node.cost.cost_usd,
                    },
                    "timestamp": node.timestamp.isoformat(),
                    "details": node.data,
                },
                "style": {
                    "background": style["background"],
                    "color": style["color"],
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "minWidth": "180px",
                },
            })
            
            # Create edges to children
            for child_id in node.children:
                edges.append({
                    "id": f"{node_id}-{child_id}",
                    "source": node_id,
                    "target": child_id,
                    "type": "smoothstep",
                    "animated": True,
                    "style": {"stroke": style["background"]},
                })
        
        return JSONResponse(content={
            "session_id": session_id,
            "root_id": tree.root_id,
            "nodes": nodes,
            "edges": edges,
            "total_nodes": tree.total_nodes,
            "max_depth": tree.max_depth,
            "total_duration_ms": tree.total_duration_ms,
            "total_cost_usd": tree.total_cost_usd,
        })
    
    except Exception as e:
        logger.error(
            "get_trajectory_tree_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/timeline")
async def get_trajectory_timeline(session_id: str) -> JSONResponse:
    """Get trajectory timeline.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Timeline events in chronological order
    """
    logger.info("get_trajectory_timeline_request", session_id=session_id)
    
    try:
        timeline = processor.to_timeline(session_id)
        
        return JSONResponse(content={
            "session_id": session_id,
            "events": [e.to_dict() for e in timeline],
            "total_events": len(timeline),
        })
    
    except Exception as e:
        logger.error(
            "get_trajectory_timeline_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/costs")
async def get_trajectory_costs(session_id: str) -> JSONResponse:
    """Get trajectory cost breakdown.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Cost breakdown by depth and type
    """
    logger.info("get_trajectory_costs_request", session_id=session_id)
    
    try:
        costs = processor.get_cost_breakdown(session_id)
        
        return JSONResponse(content=costs.to_dict())
    
    except Exception as e:
        logger.error(
            "get_trajectory_costs_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/statistics")
async def get_trajectory_statistics(session_id: str) -> JSONResponse:
    """Get trajectory execution statistics.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Execution statistics
    """
    logger.info("get_trajectory_statistics_request", session_id=session_id)
    
    try:
        stats = processor.get_statistics(session_id)
        
        return JSONResponse(content=stats.to_dict())
    
    except Exception as e:
        logger.error(
            "get_trajectory_statistics_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/export/json")
async def export_trajectory_json(session_id: str) -> JSONResponse:
    """Export trajectory as JSON.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Complete trajectory data as JSON
    """
    logger.info("export_trajectory_json_request", session_id=session_id)
    
    try:
        json_content = exporter.to_json(session_id)
        
        return JSONResponse(
            content=json.loads(json_content),
            headers={
                "Content-Disposition": f'attachment; filename="trajectory_{session_id}.json"'
            },
        )
    
    except Exception as e:
        logger.error(
            "export_trajectory_json_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/export/html")
async def export_trajectory_html(session_id: str) -> HTMLResponse:
    """Export trajectory as HTML report.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Self-contained HTML report
    """
    logger.info("export_trajectory_html_request", session_id=session_id)
    
    try:
        html_content = exporter.to_html(
            session_id,
            title=f"RLM Trajectory Report - {session_id}"
        )
        
        return HTMLResponse(
            content=html_content,
            headers={
                "Content-Disposition": f'attachment; filename="trajectory_{session_id}.html"'
            },
        )
    
    except Exception as e:
        logger.error(
            "export_trajectory_html_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/export/dot")
async def export_trajectory_dot(
    session_id: str,
    rankdir: str = Query("TB", description="Graph direction (TB, LR, BT, RL)"),
) -> PlainTextResponse:
    """Export trajectory as GraphViz DOT format.
    
    Args:
        session_id: Session identifier
        rankdir: Graph direction (TB=top-bottom, LR=left-right)
        
    Returns:
        DOT format string
    """
    logger.info("export_trajectory_dot_request", session_id=session_id, rankdir=rankdir)
    
    try:
        dot_content = exporter.to_dot(session_id, rankdir=rankdir)
        
        return PlainTextResponse(
            content=dot_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="trajectory_{session_id}.dot"'
            },
        )
    
    except Exception as e:
        logger.error(
            "export_trajectory_dot_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))
