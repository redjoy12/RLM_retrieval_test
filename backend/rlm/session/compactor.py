"""Context compactor for conversation compaction."""

from typing import List, Optional, Tuple

import structlog

from rlm.session.models import Message, Summary
from rlm.session.token_manager import TokenManager
from rlm.session.types import MessageRole, MessageType

logger = structlog.get_logger()


class ContextCompactor:
    """Compacts conversation history to manage context window.
    
    Uses pattern-based sequence detection to identify compactable
    message sequences while preserving important conversation flow.
    
    Compaction Strategy:
    1. Keep a retention window of recent messages (default: last 6)
    2. Identify compactible patterns: [Assistant] → [Tool] → [Result] → [Assistant]
    3. Summarize older sequences using a cheaper LLM
    4. Preserve user messages as anchors
    
    Example:
        ```python
        compactor = ContextCompactor(retention_window=6)
        
        # Compact if needed
        if compactor.should_compact(messages):
            compacted = await compactor.compact(messages)
        ```
    """
    
    def __init__(
        self,
        token_manager: Optional[TokenManager] = None,
        retention_window: int = 6,
        llm_client=None,  # Optional: for generating summaries
    ) -> None:
        """Initialize context compactor.
        
        Args:
            token_manager: Token manager for tracking usage
            retention_window: Number of recent messages to preserve
            llm_client: Optional LLM client for summary generation
        """
        self.token_manager = token_manager or TokenManager()
        self.retention_window = retention_window
        self.llm_client = llm_client
        
        logger.info(
            "context_compactor_initialized",
            retention_window=retention_window,
            has_llm_client=llm_client is not None,
        )
    
    def should_compact(self, messages: List[Message]) -> bool:
        """Check if messages need compaction.
        
        Args:
            messages: Current messages
            
        Returns:
            True if compaction needed
        """
        # Check if we have enough messages to compact
        if len(messages) <= self.retention_window:
            return False
        
        # Check token limits
        return self.token_manager.should_compact(messages)
    
    async def compact(
        self,
        messages: List[Message],
        session_id: str,
    ) -> Tuple[List[Message], Optional[Summary]]:
        """Compact conversation messages.
        
        Args:
            messages: All messages in conversation
            session_id: Session ID for creating summary record
            
        Returns:
            Tuple of (compacted_messages, summary_record)
        """
        if not self.should_compact(messages):
            logger.debug("no_compaction_needed", message_count=len(messages))
            return messages, None
        
        logger.info(
            "starting_compaction",
            total_messages=len(messages),
            retention_window=self.retention_window,
        )
        
        # Split into: to_compact (old) + retention (recent)
        retention_count = self.token_manager.get_retention_message_count(messages)
        to_compact = messages[:-retention_count]
        retention = messages[-retention_count:]
        
        if not to_compact:
            logger.debug("nothing_to_compact")
            return messages, None
        
        # Find compactible sequences in old messages
        sequences = self._find_compactible_sequences(to_compact)
        
        if not sequences:
            logger.debug("no_compactible_sequences_found")
            return messages, None
        
        # Generate summary for sequences
        summary_content = await self._generate_summary(sequences)
        
        # Create summary message
        summary_msg = Message(
            session_id=session_id,
            role=MessageRole.SYSTEM.value,
            content=f"[Previous conversation summary]: {summary_content}",
            tokens=self.token_manager.estimate_tokens(summary_content) + 20,  # +20 for prefix
            message_type=MessageType.SUMMARY.value,
        )
        
        # Create summary record
        summary_record = Summary(
            session_id=session_id,
            start_message_id=to_compact[0].id if to_compact[0].id else 1,
            end_message_id=to_compact[-1].id if to_compact[-1].id else len(to_compact),
            summary_content=summary_content,
            tokens_saved=self.token_manager.estimate_compaction_savings(to_compact, summary_msg),
        )
        
        # Build compacted message list: [Summary] + [Retention]
        compacted = [summary_msg] + retention
        
        logger.info(
            "compaction_complete",
            original_count=len(messages),
            compacted_count=len(compacted),
            sequences_compacted=len(sequences),
            tokens_saved=summary_record.tokens_saved,
        )
        
        return compacted, summary_record
    
    def _find_compactible_sequences(self, messages: List[Message]) -> List[List[Message]]:
        """Find compactible message sequences.
        
        Identifies patterns like:
        - [Assistant] → [Tool] → [Result] → [Assistant]
        - Multiple assistant/tool exchanges
        
        Args:
            messages: Messages to analyze
            
        Returns:
            List of compactible message sequences
        """
        sequences = []
        current_sequence = []
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # Pattern: Assistant message that might start a tool sequence
            if msg.role == MessageRole.ASSISTANT.value and not current_sequence:
                current_sequence = [msg]
                i += 1
                
                # Look for tool call pattern
                while i < len(messages):
                    next_msg = messages[i]
                    
                    # Check if it's part of the tool pattern
                    if next_msg.role in [MessageRole.TOOL.value, MessageRole.ASSISTANT.value]:
                        current_sequence.append(next_msg)
                        i += 1
                        
                        # If we hit an assistant response after tool calls, sequence is complete
                        if next_msg.role == MessageRole.ASSISTANT.value and len(current_sequence) > 1:
                            break
                    else:
                        break
                
                # Save sequence if it has the tool pattern
                if len(current_sequence) >= 3:  # Assistant → Tool(s) → Assistant
                    sequences.append(current_sequence)
                
                current_sequence = []
            else:
                i += 1
        
        return sequences
    
    async def _generate_summary(self, sequences: List[List[Message]]) -> str:
        """Generate summary of compacted sequences.
        
        Args:
            sequences: List of message sequences to summarize
            
        Returns:
            Summary text
        """
        # If no LLM client, create a basic summary
        if not self.llm_client:
            return self._generate_basic_summary(sequences)
        
        # Use LLM to generate intelligent summary
        try:
            # Build context from sequences
            context_parts = []
            for seq in sequences:
                for msg in seq:
                    if msg.role == MessageRole.ASSISTANT.value:
                        # Include first 200 chars of assistant responses
                        content = msg.content[:200]
                        context_parts.append(f"[{msg.role}]: {content}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""Summarize the following conversation exchanges concisely:

{context}

Provide a brief summary (2-3 sentences) of what was discussed or accomplished:"""
            
            # Generate summary using cheap model
            summary = await self.llm_client.generate(
                prompt,
                model="gpt-5-nano",  # Use cheap model for summaries
                max_tokens=150,
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error("summary_generation_failed", error=str(e))
            return self._generate_basic_summary(sequences)
    
    def _generate_basic_summary(self, sequences: List[List[Message]]) -> str:
        """Generate a basic summary without LLM.
        
        Args:
            sequences: Message sequences
            
        Returns:
            Basic summary
        """
        # Count different types of exchanges
        tool_calls = 0
        user_queries = 0
        
        for seq in sequences:
            for msg in seq:
                if msg.role == MessageRole.TOOL.value:
                    tool_calls += 1
                elif msg.role == MessageRole.USER.value:
                    user_queries += 1
        
        # Generate basic summary
        parts = []
        if user_queries > 0:
            parts.append(f"{user_queries} user queries")
        if tool_calls > 0:
            parts.append(f"{tool_calls} tool invocations")
        
        if parts:
            return f"Previous conversation included: {', '.join(parts)}."
        else:
            return f"Previous conversation with {len(sequences)} exchanges."
    
    def get_compaction_stats(self, messages: List[Message]) -> dict:
        """Get statistics about potential compaction.
        
        Args:
            messages: Current messages
            
        Returns:
            Statistics dictionary
        """
        total_tokens = self.token_manager.calculate_total_tokens(messages)
        usage_pct = self.token_manager.get_usage_percentage(messages)
        
        retention_count = self.token_manager.get_retention_message_count(messages)
        compactable_count = max(0, len(messages) - retention_count)
        
        return {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "usage_percentage": usage_pct,
            "retention_window": retention_count,
            "compactable_messages": compactable_count,
            "should_compact": self.should_compact(messages),
            "max_tokens": self.token_manager.max_tokens,
            "threshold_percentage": self.token_manager.warning_threshold * 100,
        }
