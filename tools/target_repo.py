"""Helper to import the *simulated-ising-machine* (target) repo on a CPU-only
machine, with stub modules for the parts that need Ascend NPUs or the C++
extensions (which are only used by the SA/HT models).

The CIM / DistIM code path used for verification does not need the real C++
extensions, so stubbing them is safe. Set ``SIM_TARGET_REPO`` to override the
default target repo location.
"""

from __future__ import annotations

import os
import sys
import types

DEFAULT_TARGET_REPO = r"c:\project\simulated-ising-machine"

# Functions imported by the target repo from its (unbuilt) C++ extensions.
_CPP_STUB_FNS = {
    "sim._sa_cpp": ["traffic_sa_step", "forward_sa_step", "standard_sa_step"],
    "sim._ht_cpp": ["ht_interact_step"],
    "sim._dist_sa_cpp": ["forward_dist_sa_step", "standard_dist_sa_step"],
}


def _install_stubs():
    for mod_name, fns in _CPP_STUB_FNS.items():
        if mod_name in sys.modules:
            continue
        mod = types.ModuleType(mod_name)
        for fn in fns:
            setattr(mod, fn, lambda *a, **k: None)
        sys.modules[mod_name] = mod

    if "torch_npu" not in sys.modules:
        tn = types.ModuleType("torch_npu")
        tn.npu = types.SimpleNamespace(
            memory_allocated=lambda *a, **k: 0,
            memory_reserved=lambda *a, **k: 0,
        )
        sys.modules["torch_npu"] = tn

    # osmnx is only used by the map-based traffic generators (network download);
    # the plain TrafficGenerator needs only networkx, so a stub is sufficient.
    if "osmnx" not in sys.modules:
        ox = types.ModuleType("osmnx")

        def _stub_fn(name):
            def _fn(*a, **k):
                raise NotImplementedError(
                    f"osmnx.{name} is stubbed; only offline TrafficGenerator "
                    "usage is supported in this environment."
                )
            return _fn

        for _name in ["load_graphml", "graph_from_place", "graph_from_point",
                      "graph_from_address", "save_graphml"]:
            setattr(ox, _name, _stub_fn(_name))
        sys.modules["osmnx"] = ox


def import_target():
    """Import the target repo's ``sim`` package, returning the module."""
    if "sim" in sys.modules:
        return sys.modules["sim"]
    _install_stubs()
    target_repo = os.environ.get("SIM_TARGET_REPO", DEFAULT_TARGET_REPO)
    if target_repo not in sys.path:
        sys.path.insert(0, target_repo)
    import sim  # noqa: F401
    return sim
