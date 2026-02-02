"""Security profiles and helpers for sandbox environments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SecurityProfile:
    """Security profile configuration for Docker sandboxes.
    
    Defines security hardening settings for container execution.
    
    Attributes:
        name: Profile name identifier
        read_only: Whether root filesystem is read-only
        cap_drop: Linux capabilities to drop
        cap_add: Linux capabilities to add (if any)
        security_opt: Security options (no-new-privileges, etc.)
        network_mode: Network isolation mode
        pids_limit: Maximum number of processes
        tmpfs_mounts: Temporary filesystem mounts
        user: User to run as (UID:GID)
    """
    
    name: str
    read_only: bool = True
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    cap_add: List[str] = field(default_factory=list)
    security_opt: List[str] = field(
        default_factory=lambda: ["no-new-privileges:true"]
    )
    network_mode: str = "none"
    pids_limit: int = 100
    tmpfs_mounts: Dict[str, str] = field(default_factory=dict)
    user: str = "nobody"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    
    def to_docker_config(self) -> Dict[str, Any]:
        """Convert profile to Docker HostConfig dictionary.
        
        Returns:
            Dictionary suitable for Docker SDK HostConfig
        """
        config = {
            "ReadonlyRootfs": self.read_only,
            "CapDrop": self.cap_drop,
            "SecurityOpt": self.security_opt,
            "NetworkMode": self.network_mode,
            "PidsLimit": self.pids_limit,
            "Memory": self._parse_memory(self.memory_limit),
            "NanoCpus": int(self.cpu_limit * 1e9),  # Convert to nanocpus
        }
        
        # Add capabilities if specified
        if self.cap_add:
            config["CapAdd"] = self.cap_add
        
        # Add tmpfs mounts
        if self.tmpfs_mounts:
            config["Tmpfs"] = self.tmpfs_mounts
        
        return config
    
    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to bytes.
        
        Args:
            memory_str: Memory string (e.g., "512m", "1g")
            
        Returns:
            Memory in bytes
        """
        memory_str = memory_str.lower().strip()
        
        multipliers = {
            "k": 1024,
            "m": 1024 ** 2,
            "g": 1024 ** 3,
            "t": 1024 ** 4,
        }
        
        for suffix, multiplier in multipliers.items():
            if memory_str.endswith(suffix):
                return int(memory_str[:-1]) * multiplier
        
        # Assume bytes if no suffix
        return int(memory_str)


class SecurityProfiles:
    """Predefined security profiles for common use cases."""
    
    @staticmethod
    def strict(
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
    ) -> SecurityProfile:
        """Strict security profile for production.
        
        Maximum isolation for untrusted code:
        - Read-only root filesystem
        - All capabilities dropped
        - No network access
        - Limited PIDs
        
        Args:
            memory_limit: Memory limit (e.g., "512m", "1g")
            cpu_limit: CPU limit in cores (e.g., 1.0, 2.0)
            
        Returns:
            SecurityProfile with strict settings
        """
        return SecurityProfile(
            name="strict",
            read_only=True,
            cap_drop=["ALL"],
            cap_add=[],
            security_opt=[
                "no-new-privileges:true",
                "seccomp:unconfined",  # Can add custom seccomp profile
            ],
            network_mode="none",
            pids_limit=50,
            tmpfs_mounts={
                "/tmp": "rw,noexec,nosuid,size=100m",
            },
            user="1000:1000",  # Non-root user
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
        )
    
    @staticmethod
    def standard(
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
    ) -> SecurityProfile:
        """Standard security profile for development.
        
        Balanced security with some flexibility:
        - Read-only root filesystem
        - All capabilities dropped
        - Bridge network (isolated but can reach internet)
        
        Args:
            memory_limit: Memory limit
            cpu_limit: CPU limit in cores
            
        Returns:
            SecurityProfile with standard settings
        """
        return SecurityProfile(
            name="standard",
            read_only=True,
            cap_drop=["ALL"],
            cap_add=[],  # No extra capabilities
            security_opt=["no-new-privileges:true"],
            network_mode="bridge",  # Allow network access
            pids_limit=100,
            tmpfs_mounts={
                "/tmp": "rw,noexec,nosuid,size=200m",
            },
            user="1000:1000",
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
        )
    
    @staticmethod
    def development(
        memory_limit: str = "1g",
        cpu_limit: float = 2.0,
    ) -> SecurityProfile:
        """Development security profile for local testing.
        
        Relaxed security for convenience:
        - Writable root filesystem
        - Network access
        - Higher resource limits
        
        Args:
            memory_limit: Memory limit
            cpu_limit: CPU limit in cores
            
        Returns:
            SecurityProfile with development settings
        """
        return SecurityProfile(
            name="development",
            read_only=False,  # Allow writes for convenience
            cap_drop=["ALL"],
            cap_add=[],  # No extra capabilities
            security_opt=["no-new-privileges:true"],
            network_mode="bridge",
            pids_limit=500,
            tmpfs_mounts={},
            user="1000:1000",
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
        )
    
    @staticmethod
    def get_profile(
        name: str,
        memory_limit: Optional[str] = None,
        cpu_limit: Optional[float] = None,
    ) -> SecurityProfile:
        """Get a security profile by name.
        
        Args:
            name: Profile name ("strict", "standard", "development")
            memory_limit: Optional override for memory limit
            cpu_limit: Optional override for CPU limit
            
        Returns:
            SecurityProfile instance
            
        Raises:
            ValueError: If profile name unknown
        """
        profiles = {
            "strict": SecurityProfiles.strict,
            "standard": SecurityProfiles.standard,
            "development": SecurityProfiles.development,
        }
        
        if name not in profiles:
            raise ValueError(
                f"Unknown security profile: {name}. "
                f"Available: {list(profiles.keys())}"
            )
        
        # Build kwargs with overrides
        kwargs = {}
        if memory_limit is not None:
            kwargs["memory_limit"] = memory_limit
        if cpu_limit is not None:
            kwargs["cpu_limit"] = cpu_limit
        
        return profiles[name](**kwargs)


class SecurityValidator:
    """Validator for security-related checks."""
    
    # Common dangerous imports
    DANGEROUS_IMPORTS = {
        "os.system",
        "os.popen",
        "os.spawn",
        "subprocess.call",
        "subprocess.run",
        "subprocess.Popen",
        "pty.spawn",
        "socket.socket",
        "urllib.request",
        "http.client",
    }
    
    # Common dangerous builtins
    DANGEROUS_BUILTINS = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "exit",
        "quit",
    }
    
    @classmethod
    def check_code_safety(
        cls,
        code: str,
        allowed_imports: Optional[List[str]] = None,
    ) -> tuple[bool, List[str]]:
        """Check code for potentially dangerous patterns.
        
        Args:
            code: Python code to check
            allowed_imports: List of allowed imports (if None, all checked)
            
        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        warnings = []
        is_safe = True
        
        # Check for dangerous imports
        for pattern in cls.DANGEROUS_IMPORTS:
            if pattern in code:
                module = pattern.split(".")[0]
                if allowed_imports is None or module not in allowed_imports:
                    warnings.append(f"Potentially dangerous import: {pattern}")
                    is_safe = False
        
        # Check for dangerous builtins
        for builtin in cls.DANGEROUS_BUILTINS:
            if builtin in code:
                warnings.append(f"Dangerous builtin usage: {builtin}")
                is_safe = False
        
        # Check for import * (wildcard imports)
        if "import *" in code:
            warnings.append("Wildcard imports are discouraged")
        
        return is_safe, warnings
    
    @classmethod
    def validate_docker_installation(cls) -> tuple[bool, str]:
        """Check if Docker is properly installed and accessible.
        
        Returns:
            Tuple of (is_installed, message)
        """
        try:
            import docker
            
            client = docker.from_env()
            version = client.version()
            
            return True, f"Docker {version['Version']} is installed and running"
            
        except ImportError:
            return False, "Docker SDK not installed (pip install docker)"
        except Exception as e:
            return False, f"Docker not accessible: {str(e)}"


def validate_security_profile(profile: SecurityProfile) -> List[str]:
    """Validate a security profile configuration.
    
    Args:
        profile: SecurityProfile to validate
        
    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []
    
    # Check for common misconfigurations
    if not profile.read_only and profile.name == "strict":
        warnings.append("Strict profile should have read_only=True")
    
    if profile.network_mode == "host" and profile.name == "strict":
        warnings.append("Strict profile should not use host network mode")
    
    if "ALL" not in profile.cap_drop:
        warnings.append("Security profile should drop ALL capabilities")
    
    # Check memory limit format
    try:
        profile._parse_memory(profile.memory_limit)
    except ValueError:
        warnings.append(f"Invalid memory limit format: {profile.memory_limit}")
    
    return warnings
