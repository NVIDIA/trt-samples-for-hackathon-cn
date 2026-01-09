# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import time

import tensorrt as trt
from tensorrt_cookbook import (MyProgressMonitor, TRTWrapperV1, build_mnist_network_trt)

def case_list():
    tw = TRTWrapperV1()
    tw.config.progress_monitor = MyProgressMonitor(False)

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

    tw.build(output_tensor_list)

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
    tw.config.progress_monitor = AnimationProgressMonitor()

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

    tw.build(output_tensor_list)

if __name__ == "__main__":
    # List each step of building engine from MyProgressMonitor
    #case_list()
    # Make a animation of the progress above as an animation
    case_animation()

    print("Finish")
