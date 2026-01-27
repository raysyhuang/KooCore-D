# src/governance/artifacts.py
"""
Governance artifacts for audit trail.

Writes governance.json every run with calibration status and health info.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os


@dataclass
class GovernanceRecord:
    """
    Governance status record for a single run.
    
    Attributes:
        calibration_used: Whether calibration model was used for ranking
        model_version: Version ID of calibration model used
        regime: Current market regime
        eligibility_passed: Whether model met eligibility requirements
        decay_detected: Whether performance decay was detected
        fallback_reason: Reason for fallback if calibration not used
        governance_flags: List of any governance warnings/flags
    """
    calibration_used: bool = False
    model_version: Optional[str] = None
    regime: Optional[str] = None
    eligibility_passed: bool = True
    decay_detected: bool = False
    fallback_reason: Optional[str] = None
    governance_flags: List[str] = field(default_factory=list)
    run_timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.run_timestamp is None:
            self.run_timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def write_governance(path: str, payload: Dict[str, Any]) -> str:
    """
    Write governance record to JSON file.
    
    Args:
        path: Output file path
        payload: Governance data dict
    
    Returns:
        Path to written file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    # Ensure timestamp is present
    if "run_timestamp" not in payload:
        payload["run_timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    
    return path


def write_governance_record(run_dir: str, date_str: str, record: GovernanceRecord) -> str:
    """
    Write governance record to standard location.
    
    Args:
        run_dir: Run output directory
        date_str: Date string (YYYY-MM-DD)
        record: GovernanceRecord dataclass
    
    Returns:
        Path to written file
    """
    path = os.path.join(run_dir, f"governance_{date_str}.json")
    return write_governance(path, record.to_dict())


def load_governance_record(path: str) -> Optional[GovernanceRecord]:
    """
    Load governance record from JSON file.
    
    Args:
        path: Path to governance JSON
    
    Returns:
        GovernanceRecord or None if file doesn't exist
    """
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return GovernanceRecord(**{
            k: v for k, v in data.items()
            if k in GovernanceRecord.__dataclass_fields__
        })
    except Exception:
        return None


class GovernanceContext:
    """
    Context manager for collecting governance information during a run.
    
    Usage:
        with GovernanceContext() as gov:
            gov.set_regime("bull")
            gov.set_calibration_used(True, "calibration_bull_2026-01-15_v1")
            # ... run logic ...
            if decay_detected:
                gov.add_flag("calibration_disabled_decay")
                gov.set_decay_detected(True)
        
        record = gov.record
    """
    
    def __init__(self):
        self._record = GovernanceRecord()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    @property
    def record(self) -> GovernanceRecord:
        return self._record
    
    def set_regime(self, regime: str) -> None:
        self._record.regime = regime
    
    def set_calibration_used(self, used: bool, model_version: Optional[str] = None) -> None:
        self._record.calibration_used = used
        self._record.model_version = model_version
    
    def set_eligibility(self, passed: bool, reason: Optional[str] = None) -> None:
        self._record.eligibility_passed = passed
        if not passed and reason:
            self._record.fallback_reason = reason
    
    def set_decay_detected(self, detected: bool) -> None:
        self._record.decay_detected = detected
        if detected:
            self.add_flag("calibration_disabled_decay")
    
    def set_fallback_reason(self, reason: str) -> None:
        self._record.fallback_reason = reason
        self._record.calibration_used = False
    
    def add_flag(self, flag: str) -> None:
        if flag not in self._record.governance_flags:
            self._record.governance_flags.append(flag)
