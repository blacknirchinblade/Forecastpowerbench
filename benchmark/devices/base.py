"""Implement the base hardware profile and its setup, verification, and restoration lifecycle.
"""

from abc import ABC, abstractmethod
from benchmark.config import DeviceCategory


class DeviceProfile(ABC):
    """Abstract interface for hardware-specific power and clock management. Implementations must ensure the device is in a locked, deterministic state before returning from setup().
    
    Bases:
        ABC
    
    Attributes:
        category: Hardware category represented by this device profile.
    """

    def __init__(self, category: DeviceCategory):
        """Initialize DeviceProfile and capture its validated configuration or runtime state.
        
        Args:
            category (DeviceCategory): Hardware category represented by this device profile.
        
        Returns:
            None; initializes the instance in place.
        """
        self.category = category

    @abstractmethod
    def setup(self) -> None:
        """Lock clocks, set CPU governor, and apply power limits. Must be called once at the start of the benchmark suite.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        pass

    @abstractmethod
    def apply_power_mode(self, mode_id: str) -> None:
        """Apply a specific power mode (e.g., '30W', 'MAXN', 'performance'). Called before each distinct power sweep if applicable.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        pass

    def verify_power_mode(self, mode_id: str) -> bool:
        """Return whether the effective hardware mode matches the requested mode.
        
        Args:
            mode_id (str): Device-specific power-mode identifier to apply or verify.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        return True

    def supports_power_mode_switching(self) -> bool:
        """Return whether supports power mode switching for DeviceProfile.
        
        Returns:
            bool: Whether the condition described by this callable holds.
        """
        return True

    @abstractmethod
    def teardown(self) -> None:
        """Restore original system states (e.g., reset CPU governor to schedutil). Must be called at the end of the benchmark suite.
        
        Returns:
            None; the operation updates state, validates input, or persists output in place.
        """
        pass
