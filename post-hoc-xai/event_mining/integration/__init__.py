"""Integration with V-Max and posthoc_xai frameworks.

Imports are deferred because these modules require JAX/V-Max.
"""


def __getattr__(name):
    if name == "VMaxAdapter":
        from event_mining.integration.vmax_adapter import VMaxAdapter
        return VMaxAdapter
    if name == "XAIBridge":
        from event_mining.integration.xai_bridge import XAIBridge
        return XAIBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VMaxAdapter", "XAIBridge"]
