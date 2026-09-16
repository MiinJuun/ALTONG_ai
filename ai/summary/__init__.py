"""Rule-based notification briefing MVP.

The models in this package are local adapter models for the summary module.  They
are intentionally kept separate from the future team-owned models in ``common``.
"""

from .service import SessionBriefingService

__all__ = ["SessionBriefingService"]

