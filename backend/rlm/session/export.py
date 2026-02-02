"""Export functionality for conversation data.

Provides export capabilities for sessions, messages, and search history
in various formats (JSON, CSV, Markdown, HTML, PDF).
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from rlm.session.manager import SessionManager
from rlm.session.types import MessageContext

logger = structlog.get_logger()


class ConversationExporter:
    """Export conversations in various formats.

    Supports multiple export formats:
    - JSON: Structured data export
    - CSV: Tabular format for analysis
    - Markdown: Human-readable format
    - HTML: Rich format with styling
    - TXT: Simple text format

    Example:
        ```python
        exporter = ConversationExporter(session_manager)

        # Export to JSON
        json_data = await exporter.export_session(session_id, format="json")

        # Export to Markdown
        md_data = await exporter.export_session(session_id, format="markdown")
        ```
    """

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        """Initialize exporter.

        Args:
            session_manager: Session manager instance
        """
        self.session_manager = session_manager or SessionManager()

        logger.info("conversation_exporter_initialized")

    async def export_session(
        self,
        session_id: str,
        format: str = "json",
        include_metadata: bool = True,
        include_search_history: bool = True,
        include_citations: bool = True,
    ) -> str:
        """Export a session in specified format.

        Args:
            session_id: Session ID
            format: Export format (json, csv, markdown, html, txt)
            include_metadata: Include session metadata
            include_search_history: Include search history
            include_citations: Include citations

        Returns:
            Exported data as string

        Raises:
            ValueError: If session not found or invalid format
        """
        # Get session data
        session = await self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Gather data
        data = await self._gather_session_data(
            session_id,
            include_metadata=include_metadata,
            include_search_history=include_search_history,
            include_citations=include_citations,
        )

        # Export based on format
        if format == "json":
            return self._export_json(data)
        elif format == "csv":
            return self._export_csv(data)
        elif format == "markdown":
            return self._export_markdown(data)
        elif format == "html":
            return self._export_html(data)
        elif format == "txt":
            return self._export_txt(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _gather_session_data(
        self,
        session_id: str,
        include_metadata: bool = True,
        include_search_history: bool = True,
        include_citations: bool = True,
    ) -> Dict[str, Any]:
        """Gather all data for a session.

        Args:
            session_id: Session ID
            include_metadata: Include metadata
            include_search_history: Include search history
            include_citations: Include citations

        Returns:
            Complete session data dictionary
        """
        session = await self.session_manager.get_session(session_id)
        messages = await self.session_manager.get_messages(session_id)
        stats = await self.session_manager.get_session_stats(session_id)

        data = {
            "session": {
                "id": session.id,
                "title": session.title,
                "status": session.status,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "last_activity": session.last_activity.isoformat()
                if session.last_activity
                else None,
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                "parent_session_id": session.parent_session_id,
                "total_tokens_used": session.total_tokens_used,
                "context_window_used": session.context_window_used,
                "default_search_strategy": session.default_search_strategy,
                "semantic_weight": session.semantic_weight,
                "keyword_weight": session.keyword_weight,
                "enable_reranking": session.enable_reranking,
                "enable_citations": session.enable_citations,
                "metadata": session.custom_metadata if include_metadata else {},
            },
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "tokens": msg.tokens,
                    "message_type": msg.message_type,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    "trajectory_id": msg.trajectory_id,
                }
                for msg in messages
            ],
            "stats": stats,
        }

        # Add search history
        if include_search_history:
            history = await self.session_manager.get_search_history(session_id, limit=1000)
            data["search_history"] = [
                {
                    "id": entry.id,
                    "query": entry.query,
                    "strategy": entry.strategy,
                    "results_count": entry.results_count,
                    "execution_time_ms": entry.execution_time_ms,
                    "semantic_weight": entry.semantic_weight,
                    "keyword_weight": entry.keyword_weight,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                }
                for entry in history
            ]

        # Add citations
        if include_citations:
            citations = await self.session_manager.get_citations(session_id)
            data["citations"] = [
                {
                    "id": cite.id,
                    "message_id": cite.message_id,
                    "chunk_id": cite.chunk_id,
                    "document_id": cite.document_id,
                    "content_snippet": cite.content_snippet,
                    "score": cite.score,
                    "created_at": cite.created_at.isoformat() if cite.created_at else None,
                }
                for cite in citations
            ]

        return data

    def _export_json(self, data: Dict[str, Any]) -> str:
        """Export to JSON format."""
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _export_csv(self, data: Dict[str, Any]) -> str:
        """Export to CSV format (messages only)."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "message_id",
                "role",
                "content",
                "tokens",
                "message_type",
                "created_at",
                "trajectory_id",
            ]
        )

        # Write messages
        for msg in data["messages"]:
            writer.writerow(
                [
                    msg["id"],
                    msg["role"],
                    msg["content"].replace('"', '""'),  # Escape quotes
                    msg["tokens"],
                    msg["message_type"],
                    msg["created_at"],
                    msg["trajectory_id"],
                ]
            )

        return output.getvalue()

    def _export_markdown(self, data: Dict[str, Any]) -> str:
        """Export to Markdown format."""
        lines = []

        # Header
        session = data["session"]
        lines.append(f"# {session['title']}\n")
        lines.append(f"**Session ID:** {session['id']}  ")
        lines.append(f"**Created:** {session['created_at']}  ")
        lines.append(f"**Status:** {session['status']}\n")

        # Messages
        lines.append("## Conversation\n")
        for msg in data["messages"]:
            role = msg["role"].capitalize()
            lines.append(f"### {role}")
            lines.append(f"*{msg['created_at']}*\n")
            lines.append(f"{msg['content']}\n")

        # Search History
        if "search_history" in data and data["search_history"]:
            lines.append("\n## Search History\n")
            for search in data["search_history"]:
                lines.append(f"- **{search['query']}** ({search['strategy']})")
                lines.append(f"  - Results: {search['results_count']}")
                lines.append(f"  - Time: {search['execution_time_ms']:.2f}ms")
                lines.append(f"  - Date: {search['created_at']}\n")

        # Citations
        if "citations" in data and data["citations"]:
            lines.append("\n## Citations\n")
            for cite in data["citations"]:
                lines.append(f"[{cite['id']}] Document {cite['document_id'][:8]}...")
                lines.append(f'   "{cite["content_snippet"][:100]}..."')
                lines.append(f"   Score: {cite['score']:.2f}\n")

        return "\n".join(lines)

    def _export_html(self, data: Dict[str, Any]) -> str:
        """Export to HTML format."""
        session = data["session"]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{session["title"]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .message {{ margin: 20px 0; padding: 15px; border-left: 4px solid #ccc; }}
        .user {{ border-left-color: #2196F3; background: #E3F2FD; }}
        .assistant {{ border-left-color: #4CAF50; background: #E8F5E9; }}
        .system {{ border-left-color: #FF9800; background: #FFF3E0; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        .search-history {{ margin-top: 30px; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
        .citation {{ margin: 10px 0; padding: 10px; background: #fffde7; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{session["title"]}</h1>
        <p><strong>Session ID:</strong> {session["id"]}</p>
        <p><strong>Created:</strong> {session["created_at"]}</p>
        <p><strong>Status:</strong> {session["status"]}</p>
    </div>
"""

        # Messages
        for msg in data["messages"]:
            role_class = msg["role"].lower()
            html += f"""
    <div class="message {role_class}">
        <div class="timestamp">{msg["created_at"]}</div>
        <strong>{msg["role"].capitalize()}</strong>
        <p>{msg["content"].replace(chr(10), "<br>")}</p>
    </div>
"""

        # Search History
        if "search_history" in data and data["search_history"]:
            html += '    <div class="search-history">\n        <h2>Search History</h2>\n'
            for search in data["search_history"]:
                html += f"""
        <p>
            <strong>{search["query"]}</strong> ({search["strategy"]})<br>
            Results: {search["results_count"]} | Time: {search["execution_time_ms"]:.2f}ms<br>
            <span class="timestamp">{search["created_at"]}</span>
        </p>
"""
            html += "    </div>\n"

        # Citations
        if "citations" in data and data["citations"]:
            html += '    <div class="citations">\n        <h2>Citations</h2>\n'
            for cite in data["citations"]:
                html += f"""
        <div class="citation">
            <strong>[{cite["id"]}]</strong> Document {cite["document_id"][:8]}...<br>
            \"{cite["content_snippet"][:100]}...\"<br>
            Score: {cite["score"]:.2f}
        </div>
"""
            html += "    </div>\n"

        html += """
</body>
</html>"""

        return html

    def _export_txt(self, data: Dict[str, Any]) -> str:
        """Export to plain text format."""
        lines = []

        session = data["session"]
        lines.append(f"Session: {session['title']}")
        lines.append(f"ID: {session['id']}")
        lines.append(f"Created: {session['created_at']}")
        lines.append(f"Status: {session['status']}")
        lines.append("=" * 60)
        lines.append("")

        for msg in data["messages"]:
            lines.append(f"[{msg['role'].upper()}] {msg['created_at']}")
            lines.append(msg["content"])
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

        return "\n".join(lines)

    async def export_batch(
        self,
        session_ids: List[str],
        format: str = "json",
        output_dir: str = "./exports",
    ) -> List[str]:
        """Export multiple sessions to files.

        Args:
            session_ids: List of session IDs
            format: Export format
            output_dir: Output directory

        Returns:
            List of exported file paths
        """
        import os

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        exported_files = []
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        for session_id in session_ids:
            try:
                data = await self.export_session(
                    session_id,
                    format=format,
                )

                # Generate filename
                session = await self.session_manager.get_session(session_id)
                safe_title = "".join(c if c.isalnum() else "_" for c in session.title)
                filename = f"{safe_title}_{session_id[:8]}_{timestamp}.{format}"
                filepath = os.path.join(output_dir, filename)

                # Write file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(data)

                exported_files.append(filepath)
                logger.info("session_exported", session_id=session_id, file=filepath)

            except Exception as e:
                logger.error("export_failed", session_id=session_id, error=str(e))

        return exported_files

    async def get_export_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of what would be exported.

        Args:
            session_id: Session ID

        Returns:
            Export summary dictionary
        """
        session = await self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        message_count = await self.session_manager.get_message_count(session_id)
        history = await self.session_manager.get_search_history(session_id)
        citations = await self.session_manager.get_citations(session_id)

        return {
            "session_id": session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "message_count": message_count,
            "search_history_count": len(history),
            "citation_count": len(citations),
            "estimated_size_bytes": message_count * 200,  # Rough estimate
            "available_formats": ["json", "csv", "markdown", "html", "txt"],
        }


# Convenience functions


async def export_session_to_file(
    session_id: str,
    filepath: str,
    format: Optional[str] = None,
    session_manager: Optional[SessionManager] = None,
) -> None:
    """Export session directly to file.

    Args:
        session_id: Session ID
        filepath: Output file path
        format: Export format (inferred from extension if None)
        session_manager: Optional session manager
    """
    # Infer format from extension if not provided
    if format is None:
        ext = filepath.split(".")[-1].lower()
        format_map = {
            "json": "json",
            "csv": "csv",
            "md": "markdown",
            "markdown": "markdown",
            "html": "html",
            "txt": "txt",
        }
        format = format_map.get(ext, "json")

    exporter = ConversationExporter(session_manager)
    data = await exporter.export_session(session_id, format=format)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data)

    logger.info("session_exported_to_file", session_id=session_id, filepath=filepath)
