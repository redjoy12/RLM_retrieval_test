"""Resource monitoring for sandbox execution."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResourceMetrics:
    """Resource usage metrics at a point in time."""
    
    timestamp: float
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    network_input_bytes: int = 0
    network_output_bytes: int = 0
    pids: int = 0


@dataclass
class ResourceUsageSummary:
    """Summary of resource usage over an execution."""
    
    execution_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    average_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    total_network_input_bytes: int = 0
    total_network_output_bytes: int = 0
    peak_pids: int = 0
    metrics_history: List[ResourceMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "execution_time_ms": self.execution_time_ms,
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "average_memory_mb": round(self.average_memory_mb, 2),
            "average_cpu_percent": round(self.average_cpu_percent, 2),
            "total_network_input_bytes": self.total_network_input_bytes,
            "total_network_output_bytes": self.total_network_output_bytes,
            "peak_pids": self.peak_pids,
        }


class ResourceMonitor:
    """Monitor resource usage of sandboxed execution.
    
    Tracks CPU, memory, network I/O, and process count over time.
    Works with both Docker containers and local processes.
    
    Example:
        ```python
        monitor = ResourceMonitor()
        
        # Start monitoring a container
        await monitor.start_monitoring("container_id")
        
        # Get current metrics
        metrics = await monitor.get_current_metrics()
        
        # Stop and get summary
        summary = await monitor.stop_monitoring()
        print(f"Peak memory: {summary.peak_memory_mb} MB")
        ```
    """
    
    def __init__(
        self,
        poll_interval: float = 1.0,
        max_history: int = 1000,
    ) -> None:
        """Initialize the resource monitor.
        
        Args:
            poll_interval: Seconds between metric polls
            max_history: Maximum number of metric samples to keep
        """
        self.poll_interval = poll_interval
        self.max_history = max_history
        
        self._metrics_history: List[ResourceMetrics] = []
        self._start_time: Optional[float] = None
        self._docker_container: Optional[Any] = None
        self._is_monitoring = False
        self._last_network_input = 0
        self._last_network_output = 0
    
    async def start_monitoring(
        self,
        container: Optional[Any] = None,
    ) -> None:
        """Start monitoring a container or process.
        
        Args:
            container: Docker container object (optional)
        """
        self._docker_container = container
        self._start_time = time.time()
        self._metrics_history = []
        self._is_monitoring = True
        self._last_network_input = 0
        self._last_network_output = 0
    
    async def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics.
        
        Returns:
            Current ResourceMetrics snapshot
        """
        if not self._is_monitoring:
            return ResourceMetrics(timestamp=time.time())
        
        if self._docker_container:
            return await self._get_docker_metrics()
        else:
            return await self._get_local_metrics()
    
    async def _get_docker_metrics(self) -> ResourceMetrics:
        """Get metrics from Docker container."""
        try:
            stats = self._docker_container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"] -
                stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"] -
                stats["precpu_stats"]["system_cpu_usage"]
            )
            
            cpu_percent = 0.0
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
            
            # Memory metrics
            memory_stats = stats.get("memory_stats", {})
            memory_usage = memory_stats.get("usage", 0)
            memory_limit = memory_stats.get("limit", 1)
            
            memory_mb = memory_usage / (1024 * 1024)
            memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            
            # Network I/O
            networks = stats.get("networks", {})
            network_input = sum(
                net.get("rx_bytes", 0) for net in networks.values()
            )
            network_output = sum(
                net.get("tx_bytes", 0) for net in networks.values()
            )
            
            # PIDs
            pids = stats.get("pids_stats", {}).get("current", 0)
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=round(cpu_percent, 2),
                memory_mb=round(memory_mb, 2),
                memory_percent=round(memory_percent, 2),
                network_input_bytes=network_input,
                network_output_bytes=network_output,
                pids=pids,
            )
            
        except Exception:
            # Return zero metrics if stats unavailable
            return ResourceMetrics(timestamp=time.time())
    
    async def _get_local_metrics(self) -> ResourceMetrics:
        """Get metrics for local process."""
        try:
            import psutil
            
            process = psutil.Process()
            with process.oneshot():
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Get system memory for percentage
                system_memory = psutil.virtual_memory()
                memory_percent = (memory_info.rss / system_memory.total) * 100
                
                # Network I/O (process-level not easily available)
                network_input = 0
                network_output = 0
                
                # PIDs (children + self)
                pids = 1 + len(process.children(recursive=True))
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=round(cpu_percent, 2),
                memory_mb=round(memory_mb, 2),
                memory_percent=round(memory_percent, 2),
                network_input_bytes=network_input,
                network_output_bytes=network_output,
                pids=pids,
            )
            
        except ImportError:
            # psutil not available
            return ResourceMetrics(timestamp=time.time())
        except Exception:
            return ResourceMetrics(timestamp=time.time())
    
    async def record_sample(self) -> ResourceMetrics:
        """Record a metrics sample.
        
        Returns:
            The recorded ResourceMetrics
        """
        metrics = await self.get_current_metrics()
        
        self._metrics_history.append(metrics)
        
        # Trim history if needed
        if len(self._metrics_history) > self.max_history:
            self._metrics_history = self._metrics_history[-self.max_history:]
        
        return metrics
    
    async def stop_monitoring(self) -> ResourceUsageSummary:
        """Stop monitoring and return usage summary.
        
        Returns:
            ResourceUsageSummary with all metrics
        """
        self._is_monitoring = False
        
        # Calculate execution time
        execution_time_ms = 0.0
        if self._start_time:
            execution_time_ms = (time.time() - self._start_time) * 1000
        
        # Calculate statistics from history
        if not self._metrics_history:
            return ResourceUsageSummary(execution_time_ms=execution_time_ms)
        
        memory_values = [m.memory_mb for m in self._metrics_history]
        cpu_values = [m.cpu_percent for m in self._metrics_history]
        pid_values = [m.pids for m in self._metrics_history]
        
        # Network totals
        if self._metrics_history:
            first = self._metrics_history[0]
            last = self._metrics_history[-1]
            net_in = last.network_input_bytes - first.network_input_bytes
            net_out = last.network_output_bytes - first.network_output_bytes
        else:
            net_in = 0
            net_out = 0
        
        summary = ResourceUsageSummary(
            execution_time_ms=round(execution_time_ms, 2),
            peak_memory_mb=max(memory_values) if memory_values else 0.0,
            peak_cpu_percent=max(cpu_values) if cpu_values else 0.0,
            average_memory_mb=sum(memory_values) / len(memory_values) if memory_values else 0.0,
            average_cpu_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            total_network_input_bytes=max(0, net_in),
            total_network_output_bytes=max(0, net_out),
            peak_pids=max(pid_values) if pid_values else 0,
            metrics_history=self._metrics_history.copy(),
        )
        
        return summary
    
    def is_monitoring(self) -> bool:
        """Check if currently monitoring.
        
        Returns:
            True if monitoring is active
        """
        return self._is_monitoring
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage so far.
        
        Returns:
            Peak memory in MB
        """
        if not self._metrics_history:
            return 0.0
        return max(m.memory_mb for m in self._metrics_history)
    
    def get_peak_cpu(self) -> float:
        """Get peak CPU usage so far.
        
        Returns:
            Peak CPU percentage
        """
        if not self._metrics_history:
            return 0.0
        return max(m.cpu_percent for m in self._metrics_history)
