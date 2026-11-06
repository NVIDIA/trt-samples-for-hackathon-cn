# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
import sys
import time

import tensorrt as trt
from tqdm import tqdm
from tensorrt_cookbook import (CookbookProgressMonitor, TRTWrapperV1, load_mnist_network_trt)

def case_list():
    tw = TRTWrapperV1()
    tw.builder_config.progress_monitor = CookbookProgressMonitor(False)

    load_mnist_network_trt(tw)

    tw.build()

class AnimationProgressMonitor(trt.IProgressMonitor):

    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True

    def phase_start(self, phase_name, parent_phase, num_steps):
        if parent_phase is not None:
            nbIndents = 1 + self._active_phases[parent_phase]["nbIndents"]
        else:
            nbIndents = 0
        self._active_phases[phase_name] = {
            "title": phase_name,
            "steps": 0,
            "num_steps": num_steps,
            "nbIndents": nbIndents,
        }
        self._redraw()

    def phase_finish(self, phase_name):
        del self._active_phases[phase_name]
        self._redraw(blank_lines=1)  # Clear the removed phase.

    def step_complete(self, phase_name, step):
        time.sleep(0.01)  # Sleep some time to see the animation
        self._active_phases[phase_name]["steps"] = step
        self._redraw()
        return self._step_result

    def _redraw(self, *, blank_lines=0):
        # The Python curses module is not widely available on Windows platforms.
        # Instead, this function uses raw terminal escape sequences. See the sample documentation for references.
        def clear_line():
            print("\x1B[2K", end="")

        def move_to_start_of_line():
            print("\x1B[0G", end="")

        def move_cursor_up(lines):
            print("\x1B[{}A".format(lines), end="")

        def progress_bar(steps, num_steps):
            INNER_WIDTH = 10
            completed_bar_chars = int(INNER_WIDTH * steps / float(num_steps))
            return "[{}{}]".format("=" * completed_bar_chars, "-" * (INNER_WIDTH - completed_bar_chars))

        # Set max_cols to a default of 200 if not run in interactive mode.
        max_cols = os.get_terminal_size().columns if sys.stdout.isatty() else 200

        move_to_start_of_line()
        for phase in self._active_phases.values():
            phase_prefix = "{indent}{bar} {title}".format(
                indent=" " * phase["nbIndents"],
                bar=progress_bar(phase["steps"], phase["num_steps"]),
                title=phase["title"],
            )
            phase_suffix = "{steps}/{num_steps}".format(**phase)
            allowable_prefix_chars = max_cols - len(phase_suffix) - 2
            if allowable_prefix_chars < len(phase_prefix):
                phase_prefix = phase_prefix[0:allowable_prefix_chars - 3] + "..."
            clear_line()
            print(phase_prefix, phase_suffix)
        for line in range(blank_lines):
            clear_line()
            print()
        move_cursor_up(len(self._active_phases) + blank_lines)
        sys.stdout.flush()

def case_animation():
    tw = TRTWrapperV1()
    tw.builder_config.progress_monitor = AnimationProgressMonitor()

    load_mnist_network_trt(tw)

    tw.build()

class TqdmProgressMonitor(trt.IProgressMonitor):
    """Draw TensorRT's build phases as nested tqdm bars.

    Adapted from `TQDMProgressMonitor` in `utilities.py` of
    https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT (Apache-2.0), where it is what a user
    watching a Stable Diffusion engine build actually sees. Nothing is copied: that code is
    TensorRT 9, and the version below fixes a defect in it (see `step_complete`).

    `AnimationProgressMonitor` above hand-draws the same thing with terminal escape sequences.
    This is the version to reach for when a progress bar library is already in the dependency tree:
    one `tqdm` per phase, indented by nesting depth.
    """

    def __init__(self, max_indent: int = 5):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        # TensorRT nests phases several levels deep and a bar per level becomes unreadable. Phases
        # deeper than this are not tracked at all -- note the early `return` in `phase_start`,
        # which is why every other method has to tolerate an unknown phase name.
        self.max_indent = max_indent

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            if parent_phase is not None:
                n_indent = self._active_phases.get(parent_phase, {}).get("n_indent", self.max_indent) + 1
                if n_indent >= self.max_indent:
                    return
                leave = False
            else:
                n_indent = 0
                leave = True  # Keep only the outermost bar on screen when the build finishes
            self._active_phases[phase_name] = {
                "tq": tqdm(total=num_steps, desc=phase_name, leave=leave, position=n_indent),
                "n_indent": n_indent,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # `phase_start` has no return value, so it cannot cancel the build. All it can do is
            # record the intent; `step_complete` is the only callback TensorRT listens to.
            self._step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases:
                phase = self._active_phases[phase_name]
                phase["tq"].update(phase["tq"].total - phase["tq"].n)  # Fill the bar before closing
                phase["tq"].close()
                # Redraw the ancestors: closing a bar can scroll them, leaving a stale picture.
                parent_phase = phase.get("parent_phase")
                while parent_phase in self._active_phases:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get("parent_phase")
                del self._active_phases[phase_name]
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases:
                phase = self._active_phases[phase_name]
                phase["tq"].update(step - phase["tq"].n)
            # The one place a build can be cancelled: returning False here aborts it, and
            # `build_serialized_network` then returns None.
            return self._step_result
        except KeyboardInterrupt:
            return False

def case_tqdm():
    """The same information as `case_animation`, drawn with tqdm instead of escape sequences."""
    tw = TRTWrapperV1()
    tw.builder_config.progress_monitor = TqdmProgressMonitor()

    load_mnist_network_trt(tw)

    tw.build()
    print(f"\nBuild finished, engine is {tw.engine_bytes.nbytes / (1 << 20):.2f} MiB")

def case_cancel():
    """Returning False from `step_complete` cancels the build, and nothing else can.

    This is the API fact the upstream implementation gets wrong. Its `phase_start` and
    `phase_finish` handlers write `_step_result = False` -- a bare name, so they assign a new local
    and the instance attribute never changes. A Ctrl-C arriving during either callback is therefore
    swallowed, and the build runs to completion. The version above writes `self._step_result`.
    """

    class CancelAfter(TqdmProgressMonitor):

        def __init__(self, n_step: int):
            super().__init__()
            self._n_step = n_step
            self._n_seen = 0

        def step_complete(self, phase_name, step):
            super().step_complete(phase_name, step)
            self._n_seen += 1
            return self._n_seen < self._n_step  # False cancels the build

    tw = TRTWrapperV1()
    tw.builder_config.progress_monitor = CancelAfter(5)

    load_mnist_network_trt(tw)

    ok = tw.build()
    print(f"\nCancelled after 5 steps: build() returned {ok}, engine_bytes is {tw.engine_bytes}")
    assert not ok and tw.engine_bytes is None, "Returning False from step_complete must cancel the build"

if __name__ == "__main__":
    # List each step of building engine from CookbookProgressMonitor
    #case_list()
    # Make a animation of the progress above as an animation
    case_animation()
    # Draw the same phases with tqdm, from Stable-Diffusion-WebUI-TensorRT
    case_tqdm()
    # Only step_complete can cancel a build
    case_cancel()

    print("Finish")
