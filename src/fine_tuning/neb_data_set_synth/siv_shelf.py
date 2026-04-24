"""Sampling helpers and rule implementations for NEB dataset synthesis."""

from __future__ import annotations

import math
from typing import Iterable

#
def allocate_quotas(
    keys: Iterable[str],
    weights: dict[str, float],
    total: int,
    capacities: dict[str, int] | None = None,
) -> dict[str, int]:
    """Allocate an integer quota across weighted buckets. It distributes <total> items across 
    the given keys according to the provided weights, (optionally) without exceeding the per-key capacity limits. 

    It is called in siv_data.py with the capacities entry fed. 

    Args:
        keys: Ordered keys to allocate across.
        weights: Relative weight for each key.
        total: Total number of items to assign.
        capacities: Optional maximum allocation per key.

    Returns:
        dict[str, int]: Integer allocation for each key.
    """

    ordered_keys = list(keys)
    remaining = max(0, int(total)) # number of items left to allocate (begins with total)
    caps = {
        key: max(0, capacities[key]) if capacities is not None else remaining
        for key in ordered_keys
    }
    # start with 0 in each bucket. 
    allocations = {key: 0 for key in ordered_keys} 

    while remaining > 0:
        eligible = [key for key in ordered_keys if caps[key] > 0] # the buckets that are not full yet. 
        if not eligible:
            break # every bucket is at capacity

        # now we need the weights of the eligible buckets. We will use these to allocate the remaining items proportionally.
        active_weights = {key: max(0.0, float(weights.get(key, 0.0))) for key in eligible}
        weight_sum = sum(active_weights.values())
        if weight_sum == 0.0:
            # If every eligible weight is zero, fall back to an even split.
            active_weights = {key: 1.0 for key in eligible}
            weight_sum = float(len(eligible))
        # this is the theoretical fractional allocation for each eligible bucket based on the active weights. 
        raw = {key: remaining * active_weights[key] / weight_sum for key in eligible}
        # use integer floor to get the guaranteed allocation for each bucket, but dont take more than capacity. 
        floors = {key: min(caps[key], int(math.floor(raw[key]))) for key in eligible}

        assigned = 0
        for key in eligible:
            if floors[key] <= 0:
                continue
            allocations[key] += floors[key] # update allocations 
            caps[key] -= floors[key] # update capacities given the assigned items
            assigned += floors[key] # track how many total items we have assigned in this round

        remaining -= assigned 
        if remaining <= 0: # Negative value will not logically happen, since assigned is based on the FLOORs of proporiotions of remaining. 
            break # done

        # Hand out leftover units by largest fractional remainder first. Tells us who is closest to recieving one 
        # more item. Sort by size of the remainder, then if necessary by weight, and finally if necessary by original key order 
        remainders = sorted(
            eligible,
            key=lambda key: (-(raw[key] - math.floor(raw[key])), -active_weights[key], ordered_keys.index(key)),
        )

        given = 0
        for key in remainders:
            if remaining <= 0: # will not be negative
                break
            if caps[key] <= 0:
                continue
            allocations[key] += 1
            caps[key] -= 1
            remaining -= 1
            given += 1

        if given == 0:
            # If every floor was zero and there are no useful remainders, give
            # the remaining slots to the highest-weight buckets that still have
            # capacity.
            fallback = sorted(eligible, key=lambda key: (-active_weights[key], ordered_keys.index(key)))
            for key in fallback:
                if remaining <= 0:
                    break
                if caps[key] <= 0:
                    continue
                allocations[key] += 1
                caps[key] -= 1
                remaining -= 1
                given += 1
            if given == 0:
                break

    return allocations


def force_energy_barrier_bias(
    catalogue: list[dict[str, object]],
    source_spec: dict[str, object],
    rule_spec: dict[str, object],
) -> list[dict[str, object]]:
    """Select frames with a bias toward barrier-region images and large forces.

    The rule works in two passes:
    1. Allocate the source quota across NEB images according to the image's
       final-energy percentile and the YAML-defined percentile bin weights.
    2. Within each chosen image, rank frames by force magnitude and energy, then
       keep the top-ranked rows.

    Args:
        catalogue: All catalogue rows available for one source.
        source_spec: Source configuration enriched with allocation metadata.
        rule_spec: Rule configuration loaded from YAML.

    Returns:
        list[dict[str, object]]: Selected catalogue rows for this source.
    """

    quota = int(source_spec["allocated_count"])
    if quota <= 0 or not catalogue:
        return []

    image_rows: dict[str, list[dict[str, object]]] = {}
    for row in catalogue:
        image_rows.setdefault(str(row["image"]), []).append(row)

    # Every row from the same image carries the same final relaxed energy, so
    # we can read it from the first row in each image group.
    final_energies = {
        image: float(rows[0]["final_image_energy"])
        for image, rows in image_rows.items()
    }
    energy_values = list(final_energies.values())
    low = min(energy_values)
    high = max(energy_values)

    image_percentiles: dict[str, float] = {}
    for image, energy in final_energies.items():
        if math.isclose(high, low):
            image_percentiles[image] = 0.5
        else:
            image_percentiles[image] = (energy - low) / (high - low)

    bins = list(rule_spec["percentile_bins"])

    def bin_weight(percentile: float) -> float:
        """Return the configured weight for an image percentile."""

        for idx, bucket in enumerate(bins):
            start = float(bucket["low"])
            end = float(bucket["high"])
            last_bucket = idx == len(bins) - 1
            if start <= percentile <= end if last_bucket else start <= percentile < end:
                return float(bucket["weight"])
        raise ValueError(f"No percentile bin covers {percentile}")

    # Convert image percentiles into image-level quotas before selecting
    # specific frames inside each image.
    image_weights = {image: bin_weight(image_percentiles[image]) for image in image_rows}
    image_capacities = {image: len(rows) for image, rows in image_rows.items()}
    image_allocations = allocate_quotas(image_rows.keys(), image_weights, quota, image_capacities)

    selected: list[dict[str, object]] = []
    for image in source_spec["images"]:
        image_name = str(image)
        rows = image_rows.get(image_name, [])
        if not rows:
            continue
        chosen = image_allocations.get(image_name, 0)
        ranked = sorted(
            rows,
            key=lambda row: (
                # Favor hard configurations first: large forces, then higher
                # energy, then earlier steps as a stable tie-break.
                -float(row["max_atom_force"]),
                -float(row["frame_energy"]),
                int(row["ionic_step"]),
            ),
        )
        for row in ranked[:chosen]:
            selected_row = dict(row)
            selected_row["image_percentile"] = image_percentiles[image_name]
            selected_row["image_bin_weight"] = image_weights[image_name]
            selected_row["image_selected_quota"] = chosen
            selected.append(selected_row)

    return selected


# Rule registry used by the main pipeline to resolve YAML rule names.
RULES = {
    "force_energy_barrier_bias": force_energy_barrier_bias,
}
