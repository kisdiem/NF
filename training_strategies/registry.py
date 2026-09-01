"""External parameter registry; model classes remain unmodified."""
from dataclasses import dataclass


@dataclass
class ParameterInventory:
    model_key: str
    synaptic_names: list
    intrinsic_names: list
    other_names: list
    discrete_intrinsic_names: list

    def parameters(self, model, group):
        names = set(getattr(self, f"{group}_names"))
        return [p for n, p in model.named_parameters() if n in names and p.requires_grad]

    def validate(self, model):
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        groups = set(self.synaptic_names) | set(self.intrinsic_names) | set(self.other_names)
        missing = trainable - groups
        overlap = ((set(self.synaptic_names) & set(self.intrinsic_names)) |
                   (set(self.synaptic_names) & set(self.other_names)) |
                   (set(self.intrinsic_names) & set(self.other_names)))
        if missing or overlap:
            raise ValueError(f"invalid inventory: missing={sorted(missing)}, overlap={sorted(overlap)}")


def _intrinsic_for(model_key, name):
    if model_key == "minimal_local_nf":
        return name == "field.decay_raw"
    if model_key.startswith("local_electrical"):
        return name.startswith("field.") and any(x in name for x in (
            "theta_raw", "strength_raw", "decay_raw", "sign_raw",
            "rho_raw", "beta_raw", "gamma_raw"))
    if model_key in {"dynamic_nf", "hierarchical_nf"}:
        return (("gain" in name and name.endswith("_raw")) or
                name == "field.relation_gain_raw" or
                name in {"field.branch_bias", "field.bias1", "field.bias2"})
    if model_key == "bio_neuron":
        return name.startswith("bio.") and any(x in name for x in (
            "branch_bias", "branch_gain_raw", "soma_gain_raw",
            "theta_raw", "adaptation_raw"))
    if model_key == "discrete_nf_v3":
        return name.endswith("field.T") or name.endswith("field.s_param")
    if model_key == "directional_rect_v4":
        return name.endswith("field.theta") or name.endswith("field.g_raw")
    return False


def classify_parameters(model, model_key):
    synaptic, intrinsic, other = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _intrinsic_for(model_key, name):
            intrinsic.append(name)
        elif any(x in name for x in ("weight", "bias", "kernel", "proj", "relation",
                                     "Q", "A", ".c", ".w", ".b", "eta",
                                     "branch_mix", "mix", "exc_", "inh_",
                                     "column_attr", "full_raw", "energy_score")):
            synaptic.append(name)
        else:
            other.append(name)
    discrete = []
    if model_key == "discrete_nf_v3":
        discrete = [n for n, _ in model.named_buffers()
                    if n.endswith("field.L") or n.endswith("field.D")]
    inv = ParameterInventory(model_key, synaptic, intrinsic, other, discrete)
    inv.validate(model)
    return inv
