"""CBM-V1: Hard Concept Bottleneck Model for V-Max SAC.

Architecture:
    observation → pretrained LQ encoder → z → concept_head → c → actor/critic

Modules:
    config          CBMConfig dataclass
    networks        CBMPolicyNetwork, CBMValueNetwork, ConceptHead
    concept_loss    Masked concept supervision loss (BCE + Huber)
    cbm_sac_factory CBM-aware SAC factory with concept loss integration
"""

from cbm_v1.config import CBMConfig
from cbm_v1.concept_loss import concept_loss
from cbm_v1.networks import CBMPolicyNetwork, CBMValueNetwork, ConceptHead
from cbm_v1.cbm_sac_factory import (
    CBMSACNetworks,
    CBMSACNetworkParams,
    CBMSACTrainingState,
    initialize,
    make_inference_fn,
    make_networks,
    make_sgd_step,
    set_cbm_policy_module,
)

__all__ = [
    "CBMConfig",
    "concept_loss",
    "CBMPolicyNetwork",
    "CBMValueNetwork",
    "ConceptHead",
    "CBMSACNetworks",
    "CBMSACNetworkParams",
    "CBMSACTrainingState",
    "initialize",
    "make_inference_fn",
    "make_networks",
    "make_sgd_step",
    "set_cbm_policy_module",
]
