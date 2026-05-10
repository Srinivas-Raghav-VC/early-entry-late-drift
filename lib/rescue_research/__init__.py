"""
Rescue Research — Modular, research-grade pipeline for cross-script rescue.

This package reimplements the experimental pipeline with:
- Single config (no inconsistent defaults)
- Three-way split for any selection step (no selection bias)
- Causal mediation (NIE/NDE) in the main pipeline
- One primary outcome, clearly defined

Uses the parent codebase (fresh_experiments) for model/transcoder loading
and patching mechanics; this package provides the correct orchestration.
"""

__version__ = "0.1.0"
