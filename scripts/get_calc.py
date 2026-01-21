



def get_calc_object(model):
    if model == "small-omat-0":
        from mace.calculators import mace_mp
        calc = mace_mp(model=model, device="cuda", default_dtype="float64")
    if model == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(device="cuda")
    if model == "orb": 
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        device="cuda"
        # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
        orbff = pretrained.orb_v3_conservative_inf_omat(
        device=device,
        precision="float32-high",   # or "float32-highest" / "float64
        )
        calc = ORBCalculator(orbff, device=device)
        print("using Orb calculator")
    if model == "pet-mad":
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(model="pet-mad-s", version="1.1.0", device="cpu", non_conservative=True)
        print("using pet-mad calculator")
    if model == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        calc = CHGNetCalculator()
        print("using chgnet calc")
    return calc





"""

import os
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
models_root = repo_root / "assets" / "models"

def get_calc_object(calculator: str, device="cuda"):
    if calculator == "small-omat-0":
        from mace.calculators import mace_mp
        return mace_mp(
            model=str(models_root / "mace-small-omat-0.model"),
            device=device,
            default_dtype="float64",
        )

    if calculator == "mattersim":
        PIN = models_root / "mattersim"
        os.environ["XDG_CACHE_HOME"] = str(PIN)
        os.environ["TORCH_HOME"] = str(PIN / "torch")

        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=device)

    if calculator == "orb":
        PIN = models_root / "orb-v3-conservative"
        os.environ["XDG_CACHE_HOME"] = str(PIN)
        os.environ["TORCH_HOME"] = str(PIN / "torch")

        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",
        )
        return ORBCalculator(orbff, device=device)

    if calculator == "pet-mad":
        PIN = models_root / "pet-mad-s-v1.1.0"
        os.environ["XDG_CACHE_HOME"] = str(PIN)

        from upet.calculator import UPETCalculator
        return UPETCalculator(
            model="pet-mad-s",
            version="1.1.0",
            device="cpu",
            non_conservative=True,
        )

    if calculator == "chgnet":
        PIN = models_root / "chgnet"
        os.environ["XDG_CACHE_HOME"] = str(PIN)
        os.environ["TORCH_HOME"] = str(PIN / "torch")

        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator()

    raise ValueError(f"Unknown calculator '{calculator}'")

"""