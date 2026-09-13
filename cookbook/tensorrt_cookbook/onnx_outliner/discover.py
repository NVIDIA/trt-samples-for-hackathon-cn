# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""P2 path A -- serially repeated blocks as repeated substrings.

Once the DAG is flattened with the canonical order from ordering.py, a stack of
identical blocks becomes a periodic string, so "find the repeated module"
becomes "find the repeated substring". On a 6-layer TransformerEncoder this
finds period 41 repeating 6 times and covers 246 of 246 nodes.

Scoring is MDL-flavoured rather than frequency based, because plain frequency is
degenerate: when a 41-node block occurs 6 times, *every* sub-pattern of it also
occurs 6 times. What we want is the pattern that compresses the most.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

_HASH_BASE = 1_000_003
_HASH_MOD = (1 << 61) - 1

@dataclass
class Candidate:
    """A repeated substring: `length` labels starting at each of `position_list`."""

    length: int
    position_list: list[int]
    label_tuple: tuple

    @property
    def gain(self) -> int:
        """Nodes removed from the main graph, minus the one call node kept per instance.

        `(length - 1) * (n_instance - 1)` is the classic MDL substructure gain:
        every instance collapses to a single node, and one copy of the body has
        to be paid for.
        """
        return (self.length - 1) * (len(self.position_list) - 1)

def _greedy_non_overlapping(position_list: list[int], length: int) -> list[int]:
    """Keep the left-most set of occurrences that do not overlap."""
    kept, last = [], None
    for position in position_list:
        if last is None or position - last >= length:
            kept.append(position)
            last = position
    return kept

def _rolling_hash(code_list: list[int], length: int) -> list[int]:
    """Hash of every window of `length` consecutive codes, in O(n)."""
    if length > len(code_list):
        return []
    power = pow(_HASH_BASE, length - 1, _HASH_MOD)
    value, hash_list = 0, []
    for i, code in enumerate(code_list):
        value = (value * _HASH_BASE + code) % _HASH_MOD
        if i >= length:
            value = (value - code_list[i - length] * power * _HASH_BASE) % _HASH_MOD
        if i >= length - 1:
            hash_list.append(value)
    return hash_list

def find_candidates(label_list: list, min_repeat: int, min_size: int) -> list[Candidate]:
    """Best candidate at every usable length, sorted by decreasing gain.

    One pass per length with a rolling hash, so the whole scan is
    O(n^2 / min_repeat) hash operations rather than O(n^2 * length) tuple builds.

    Documented limitation: only the single highest-gain bucket per length is
    turned into a candidate. A length that hides two different repeated blocks
    therefore reports only the better one on this pass. That is not a silent
    cap: the caller masks committed instances and calls this again, so the
    second block is found on the next round.
    """
    code_of, code_list = {}, []
    for label in label_list:
        code_list.append(code_of.setdefault(label, len(code_of) + 1))

    candidate_list = []
    for length in range(min_size, len(code_list) // min_repeat + 1):
        bucket = defaultdict(list)
        for position, value in enumerate(_rolling_hash(code_list, length)):
            bucket[value].append(position)

        # Pick the most promising bucket by hash count first. Materialising the
        # real label tuples for *every* bucket would cost O(n * length) per
        # length, which dominates everything else on a few-thousand-node graph.
        best_kept, best_gain = None, 0
        for position_list in bucket.values():
            if len(position_list) < min_repeat:
                continue
            kept = _greedy_non_overlapping(position_list, length)
            gain = (length - 1) * (len(kept) - 1)
            if len(kept) >= min_repeat and gain > best_gain:
                best_kept, best_gain = kept, gain
        if best_kept is None:
            continue

        # Confirm against the real labels. A 61-bit hash makes a collision
        # essentially impossible, but "essentially" is not "never", and a
        # collision must cost accuracy of the search, never correctness.
        label_tuple = tuple(label_list[best_kept[0]:best_kept[0] + length])
        confirmed = [p for p in best_kept if tuple(label_list[p:p + length]) == label_tuple]
        if len(confirmed) >= min_repeat:
            candidate_list.append(Candidate(length, confirmed, label_tuple))

    return sorted(candidate_list, key=lambda c: (-c.gain, -c.length))
