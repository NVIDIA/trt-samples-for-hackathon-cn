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

from pathlib import Path

import tensorrt as trt
from tensorrt_cookbook import APIExcludeSet, grep_used_members

# trt.ISymExpr related
sym_expr = trt.ISymExpr()
print(f"{sym_expr = }")
# public_member = APIExcludeSet.analyze_public_members(trt.ISymExpr(), b_print=True)
# grep_used_members(Path(__file__), public_member)
# sym_expr.dtype()
# sym_expr.expr()
# sym_expr.type()

# trt.ISymExprs related
# trt.ISymExprs has no constructor
public_member = APIExcludeSet.analyze_public_members(obj_class=trt.ISymExprs, b_print=True)
grep_used_members(Path(__file__), public_member)
print(f"{trt.ISymExprs.nbSymExprs = }")

# trt.IDimensionExpr related
# trt.IDimensionExpr has no constructor
public_member = APIExcludeSet.analyze_public_members(obj_class=trt.IDimensionExpr, b_print=True)
grep_used_members(Path(__file__), public_member)
# trt.IDimensionExpr.get_constant_value()
# trt.IDimensionExpr.is_constant()
# trt.IDimensionExpr.is_size_tensor()

# trt.DimsExprs related
dim_exps = trt.DimsExprs(2)
public_member = APIExcludeSet.analyze_public_members(obj_class=trt.DimsExprs, b_print=True)
grep_used_members(Path(__file__), public_member)

print(f"{dim_exps[0] = }, {dim_exps[1] = }")

print("Finish")
