"""Token manager for tracking context window usage."""

from typing import List, Optional

import structlog

from rlm.session.models import Message
from rlm.session.types import MessageContext

logger = structlog.get_logger()


class TokenManager:
    """Manages token usage and context window limits.
    
    Tracks token consumption across sessions and determines when
    context compaction is needed.
    
    Example:
        ```python
        manager = TokenManager(max_tokens=128000)
        
        # Check if compaction needed
        if await manager.should_compact(messages):
            # Trigger compaction
            pass
        
        # Estimate tokens for text
        tokens = manager.estimate_tokens("Hello world")
        ```
    """
    
    def __init__(
        self,
        max_tokens: int = 128000,
        warning_threshold: float = 0.8,
        retention_tokens: int = 6000,  # ~6K tokens for retention window
    ) -> None:
        """Initialize token manager.
        
        Args:
            max_tokens: Maximum context window size
            warning_threshold: Threshold (0-1) to trigger warnings
            retention_tokens: Minimum tokens to always retain
        """
        self.max_tokens = max_tokens
        self.warning_threshold = warning_threshold
        self.retention_tokens = retention_tokens
        self.compaction_threshold = int(max_tokens * warning_threshold)
        
        logger.info(
            "token_manager_initialized",
            max_tokens=max_tokens,
            warning_threshold=warning_threshold,
            compaction_threshold=self.compaction_threshold,
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses a rough approximation of 4 characters per token.
        For production, consider using tiktoken for accurate counts.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimate: ~4 characters per token for English
        # This is a conservative estimate
        return len(text) // 4 + 1
    
    def calculate_message_tokens(self, message: Message) -> int:
        """Calculate tokens for a message including metadata.
        
        Args:
            message: Message to calculate
            
        Returns:
            Token count
        """
        # Content tokens
        content_tokens = self.estimate_tokens(message.content)
        
        # Role and formatting overhead (~4 tokens)
        overhead = 4
        
        return content_tokens + overhead
    
    def calculate_total_tokens(self, messages: List[Message]) -> int:
        """Calculate total tokens for a list of messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            if msg.tokens and msg.tokens > 0:
                # Use stored token count if available
                total += msg.tokens
            else:
                # Estimate if not stored
                total += self.calculate_message_tokens(msg)
        
        return total
    
    def should_compact(self, messages: List[Message]) -> bool:
        """Determine if context should be compacted.
        
        Args:
            messages: Current conversation messages
            
        Returns:
            True if compaction needed
        """
        total_tokens = self.calculate_total_tokens(messages)
        
        should_compact = total_tokens > self.compaction_threshold
        
        if should_compact:
            logger.info(
                "compaction_triggered",
                total_tokens=total_tokens,
                threshold=self.compaction_threshold,
                max_tokens=self.max_tokens,
            )
        
        return should_compact
    
    def get_remaining_tokens(self, messages: List[Message]) -> int:
        """Get remaining tokens before hitting limit.
        
        Args:
            messages: Current messages
            
        Returns:
            Remaining token budget
        """
        used = self.calculate_total_tokens(messages)
        return max(0, self.max_tokens - used)
    
    def get_usage_percentage(self, messages: List[Message]) -> float:
        """Get context window usage percentage.
        
        Args:
            messages: Current messages
            
        Returns:
            Usage as percentage (0-100)
        """
        used = self.calculate_total_tokens(messages)
        return (used / self.max_tokens) * 100
    
    def is_near_limit(self, messages: List[Message]) -> bool:
        """Check if approaching context limit.
        
        Args:
            messages: Current messages
            
        Returns:
            True if near limit
        """
        return self.get_usage_percentage(messages) >= (self.warning_threshold * 100)
    
    def estimate_compaction_savings(
        self,
        messages_to_compact: List[Message],
        summary_message: Message,
    ) -> int:
        """Estimate tokens saved by compaction.
        
        Args:
            messages_to_compact: Messages that will be compacted
            summary_message: Summary that will replace them
            
        Returns:
            Estimated tokens saved
        """
        original_tokens = self.calculate_total_tokens(messages_to_compact)
        summary_tokens = self.calculate_message_tokens(summary_message)
        
        return max(0, original_tokens - summary_tokens)
    
    def get_retention_message_count(self, messages: List[Message]) -> int:
        """Determine how many recent messages to retain.
        
        Keeps at least the last N messages or until retention_tokens is met,
        whichever is larger.
        
        Args:
            messages: All messages
            
        Returns:
            Number of messages to retain from the end
        """
        min_retention = 6  # Always keep last 6 messages minimum
        
        if len(messages) <= min_retention:
            return len(messages)
        
        # Count backwards until we hit retention_tokens
        retained = 0
        retained_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = msg.tokens or self.calculate_message_tokens(msg)
            
            if retained_tokens + msg_tokens > self.retention_tokens and retained >= min_retention:
                break
            
            retained_tokens += msg_tokens
            retained += 1
        
        return max(retained, min_retention)
