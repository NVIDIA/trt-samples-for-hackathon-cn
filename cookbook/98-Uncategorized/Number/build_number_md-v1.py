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

import math
from typing import Dict
from dataclasses import dataclass
from enum import Enum

class FloatFormat(Enum):
    """支持的浮点格式枚举"""
    FP32 = "FP32"  # IEEE 754 single precision
    FP16 = "FP16"  # IEEE 754 half precision
    BF16 = "BF16"  # Brain Floating Point
    TF32 = "TF32"  # TensorFloat-32 (19-bit)
    FP64 = "FP64"  # IEEE 754 double precision
    FP8_E4M3 = "FP8-E4M3"  # FP8 E4M3 format
    FP8_E5M2 = "FP8-E5M2"  # FP8 E5M2 format
    FP8_E8M0 = "FP8-E8M0"  # FP8 E8M0 format (unsigned, exponent only)
    FP6_E3M2 = "FP6-E3M2"  # FP6 E3M2 format
    FP6_E2M3 = "FP6-E2M3"  # FP6 E2M3 format
    FP4_E2M1 = "FP4-E2M1"  # FP4 E2M1 format
    FP4_E0M3 = "FP4-E0M3"  # FP4 E0M3 format (no exponent, only mantissa)

@dataclass
class FloatFormatSpec:
    """浮点格式规格"""
    name: str
    total_bits: int
    sign_bits: int
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    has_implicit_bit: bool = True
    supports_inf: bool = True
    supports_nan: bool = True
    is_unsigned: bool = False
    is_exponent_only: bool = False
    is_mantissa_only: bool = False

FORMAT_SPECS = {
    FloatFormat.FP32: FloatFormatSpec("FP32", 32, 1, 8, 23, 127),
    FloatFormat.FP16: FloatFormatSpec("FP16", 16, 1, 5, 10, 15),
    FloatFormat.BF16: FloatFormatSpec("BF16", 16, 1, 8, 7, 127),
    FloatFormat.TF32: FloatFormatSpec("TF32", 19, 1, 8, 10, 127),
    FloatFormat.FP64: FloatFormatSpec("FP64", 64, 1, 11, 52, 1023),
    FloatFormat.FP8_E4M3: FloatFormatSpec("FP8_E4M3", 8, 1, 4, 3, 7, supports_inf=False, supports_nan=True),
    FloatFormat.FP8_E5M2: FloatFormatSpec("FP8_E5M2", 8, 1, 5, 2, 15),
    FloatFormat.FP8_E8M0: FloatFormatSpec("FP8_E8M0", 8, 0, 8, 0, 127, supports_inf=False, supports_nan=True, is_unsigned=True, is_exponent_only=True, has_implicit_bit=False),
    FloatFormat.FP6_E3M2: FloatFormatSpec("FP6_E3M2", 6, 1, 3, 2, 3, supports_inf=False, supports_nan=False),
    FloatFormat.FP6_E2M3: FloatFormatSpec("FP6_E2M3", 6, 1, 2, 3, 1, supports_inf=False, supports_nan=False),
    FloatFormat.FP4_E2M1: FloatFormatSpec("FP4_E2M1", 4, 1, 2, 1, 1, supports_inf=False, supports_nan=False),
    FloatFormat.FP4_E0M3: FloatFormatSpec("FP4_E0M3", 4, 1, 0, 3, 0, supports_inf=False, supports_nan=False, is_mantissa_only=True, has_implicit_bit=False),
}  # wili，根据现有的分支条件确认一下

class UniversalFloatConverter:
    """通用浮点格式转换器"""

    def __init__(self):
        self.format_specs = FORMAT_SPECS
        # FP4-E0M3的查找表，基于论文中的量化表
        self.fp4_e0m3_lookup = {0: -6.0, 1: -4.0, 2: -3.0, 3: -2.0, 4: -1.5, 5: -1.0, 6: -0.5, 7: 0.0}
        self.fp4_e0m3_reverse_lookup = {v: k for k, v in self.fp4_e0m3_lookup.items()}

    def binary_string_to_float(self, binary_string: str, format_type: FloatFormat) -> float:
        """
        将二进制字符串转换为浮点数

        参数:
            binary_string: 二进制字符串
            format_type: 浮点格式类型

        返回:
            float: 对应的浮点数值
        """
        spec = self.format_specs[format_type]

        # 验证输入
        if len(binary_string) != spec.total_bits:
            raise ValueError(f"{spec.name} requires {spec.total_bits} bits, got {len(binary_string)}")

        if not all(c in '01' for c in binary_string):
            raise ValueError("Binary string must contain only 0 and 1")

        # 处理特殊格式
        if spec.is_exponent_only:
            return self._e8m0_to_float(binary_string, spec)
        elif spec.is_mantissa_only:
            return self._e0m3_to_float(binary_string, spec)
        else:
            return self._standard_to_float(binary_string, spec)

    def _e8m0_to_float(self, binary_string: str, spec: FloatFormatSpec) -> float:
        """处理FP8-E8M0格式（纯指数，无符号）"""
        # E8M0: 8位全为指数位，无符号位和尾数位
        exponent = int(binary_string, 2)

        # 特殊值处理
        if spec.supports_nan and exponent == 255:  # 0xFF 表示 NaN
            return float('nan')

        # 减去偏移量得到真实指数
        true_exponent = exponent - spec.exponent_bias

        # 计算值：2^true_exponent
        return 2.0 ** true_exponent

    def _e0m3_to_float(self, binary_string: str, spec: FloatFormatSpec) -> float:
        """处理FP4-E0M3格式（纯尾数，无指数）"""
        # E0M3: 1位符号位 + 3位尾数位，无指数位
        sign_bit = int(binary_string[0], 2)
        mantissa = int(binary_string[1:], 2)  # 3位尾数

        # 使用查找表获取值
        base_value = self.fp4_e0m3_lookup.get(mantissa, 0.0)

        # 应用符号
        return -base_value if sign_bit else base_value

    def _standard_to_float(self, binary_string: str, spec: FloatFormatSpec) -> float:
        """处理标准浮点格式"""
        # 解析二进制字符串
        sign_bit = int(binary_string[0], 2)
        exponent_bits = binary_string[1:1 + spec.exponent_bits]
        mantissa_bits = binary_string[1 + spec.exponent_bits:]

        exponent = int(exponent_bits, 2)
        mantissa = int(mantissa_bits, 2) if mantissa_bits else 0

        # 处理特殊情况
        if spec.supports_inf and spec.supports_nan:
            # 标准IEEE 754行为
            if exponent == (1 << spec.exponent_bits) - 1:  # 指数全1
                if mantissa == 0:
                    return float('inf') if sign_bit == 0 else float('-inf')
                else:
                    return float('nan')
        elif spec.supports_nan and not spec.supports_inf:
            # FP8 E4M3格式：只有NaN，没有Inf
            if exponent == (1 << spec.exponent_bits) - 1 and mantissa != 0:
                return float('nan')

        # 计算符号
        sign = -1 if sign_bit else 1

        # 处理不同类型的数值
        if exponent == 0:
            # 零或次正规数
            if mantissa == 0:
                return 0.0 if sign_bit == 0 else -0.0
            else:
                # 次正规数
                if spec.has_implicit_bit:
                    significand = mantissa / (1 << spec.mantissa_bits)
                    return sign * significand * (2 ** (1 - spec.exponent_bias))
                else:
                    significand = mantissa / (1 << spec.mantissa_bits)
                    return sign * significand * (2 ** (1 - spec.exponent_bias))
        else:
            # 正规数
            if spec.has_implicit_bit:
                significand = 1 + mantissa / (1 << spec.mantissa_bits)
            else:
                significand = mantissa / (1 << spec.mantissa_bits)

            exponent_value = exponent - spec.exponent_bias
            return sign * significand * (2 ** exponent_value)

    def float_to_binary_string(self, value: float, format_type: FloatFormat) -> str:
        """
        将浮点数转换为二进制字符串

        参数:
            value: 浮点数值
            format_type: 目标浮点格式类型

        返回:
            str: 二进制字符串
        """
        spec = self.format_specs[format_type]

        # 处理特殊格式
        if spec.is_exponent_only:
            return self._float_to_e8m0(value, spec)
        elif spec.is_mantissa_only:
            return self._float_to_e0m3(value, spec)
        else:
            return self._float_to_standard(value, spec)

    def _float_to_e8m0(self, value: float, spec: FloatFormatSpec) -> str:
        """将浮点数转换为FP8-E8M0格式"""
        # 处理特殊值
        if math.isnan(value):
            if spec.supports_nan:
                return '11111111'  # 0xFF 表示 NaN
            else:
                return '01111111'  # 最大可表示值

        # 处理负数 - E8M0是无符号格式，取绝对值
        abs_value = abs(value)

        # 处理零值
        if abs_value == 0.0:
            return '00000000'  # 2^-127

        # 找到最接近的2的幂次方
        if abs_value >= 1:
            true_exponent = int(round(math.log2(abs_value)))
        else:
            true_exponent = int(round(math.log2(abs_value)))

        # 限制指数范围 [-127, 127]
        true_exponent = max(-127, min(127, true_exponent))

        # 加上偏移量
        biased_exponent = true_exponent + spec.exponent_bias

        # 确保在有效范围内 [0, 254]
        biased_exponent = max(0, min(254, biased_exponent))

        return format(biased_exponent, '08b')

    def _float_to_e0m3(self, value: float, spec: FloatFormatSpec) -> str:
        """将浮点数转换为FP4-E0M3格式"""
        # 处理零值
        if value == 0.0:
            return '00111'  # 0.0 的编码

        # 处理符号
        sign_bit = '1' if value < 0 else '0'
        abs_value = abs(value)

        # 在查找表中找到最接近的值
        closest_value = None
        closest_distance = float('inf')
        closest_mantissa = 7  # 默认为0.0

        for mantissa, table_value in self.fp4_e0m3_lookup.items():
            distance = abs(abs_value - abs(table_value))
            if distance < closest_distance:
                closest_distance = distance
                closest_value = table_value
                closest_mantissa = mantissa

        # 如果最接近的值是负数，需要调整符号位
        if closest_value and closest_value < 0:
            sign_bit = '1'

        mantissa_bits = format(closest_mantissa, '03b')
        return sign_bit + mantissa_bits

    def _float_to_standard(self, value: float, spec: FloatFormatSpec) -> str:
        """将浮点数转换为标准浮点格式"""
        # 处理特殊值
        if math.isnan(value):
            if not spec.supports_nan:
                # 如果不支持NaN，返回最大正规数
                return self._get_max_normal_binary(spec)

            # 构造NaN
            sign_bit = '0'
            exponent_max = (1 << spec.exponent_bits) - 1
            exponent_bits = format(exponent_max, f'0{spec.exponent_bits}b')
            mantissa_bits = '1' + '0' * (spec.mantissa_bits - 1)  # 至少一个1表示NaN
            return sign_bit + exponent_bits + mantissa_bits

        if math.isinf(value):
            if not spec.supports_inf:
                # 如果不支持Inf，返回最大正规数
                return self._get_max_normal_binary(spec)

            # 构造无穷大
            sign_bit = '1' if value < 0 else '0'
            exponent_max = (1 << spec.exponent_bits) - 1
            exponent_bits = format(exponent_max, f'0{spec.exponent_bits}b')
            mantissa_bits = '0' * spec.mantissa_bits
            return sign_bit + exponent_bits + mantissa_bits

        # 处理零
        if value == 0.0:
            sign_bit = '1' if math.copysign(1, value) < 0 else '0'
            return sign_bit + '0' * (spec.total_bits - 1)

        # 处理正规数
        abs_value = abs(value)
        sign_bit = '1' if value < 0 else '0'

        # 计算指数和尾数
        if abs_value >= 1:
            exponent = int(math.floor(math.log2(abs_value)))
        else:
            exponent = int(math.ceil(math.log2(abs_value))) if abs_value > 0 else -spec.exponent_bias

        # 限制指数范围
        max_exponent = (1 << spec.exponent_bits) - 2 if (spec.supports_inf or spec.supports_nan) else (1 << spec.exponent_bits) - 1
        min_exponent = 1 - spec.exponent_bias

        if exponent > max_exponent:
            # 溢出，返回最大可表示数
            return self._get_max_normal_binary(spec)
        elif exponent < min_exponent:
            # 下溢，处理为次正规数或零
            if spec.has_implicit_bit:
                # 尝试次正规数表示
                significand = abs_value / (2 ** (1 - spec.exponent_bias))
                if significand < 1:
                    significand *= (1 << spec.mantissa_bits)
                    mantissa = int(round(significand))
                    if mantissa == 0:
                        return sign_bit + '0' * (spec.total_bits - 1)  # 零
                    else:
                        exponent_bits = '0' * spec.exponent_bits
                        mantissa_bits = format(mantissa, f'0{spec.mantissa_bits}b')
                        return sign_bit + exponent_bits + mantissa_bits
                else:
                    return sign_bit + '0' * (spec.total_bits - 1)  # 零
            else:
                return sign_bit + '0' * (spec.total_bits - 1)  # 零

        # 正规数
        exponent_biased = exponent + spec.exponent_bias
        exponent_bits = format(exponent_biased, f'0{spec.exponent_bits}b')

        # 计算尾数
        if spec.has_implicit_bit:
            significand = abs_value / (2 ** exponent)
            significand -= 1  # 去掉隐含的1
            mantissa = int(round(significand * (1 << spec.mantissa_bits)))
        else:
            significand = abs_value / (2 ** exponent)
            mantissa = int(round(significand * (1 << spec.mantissa_bits)))

        # 限制尾数范围
        mantissa = max(0, min(mantissa, (1 << spec.mantissa_bits) - 1))
        mantissa_bits = format(mantissa, f'0{spec.mantissa_bits}b')

        return sign_bit + exponent_bits + mantissa_bits

    def _get_max_normal_binary(self, spec: FloatFormatSpec) -> str:
        """获取最大正规数的二进制表示"""
        if spec.supports_inf or spec.supports_nan:
            max_exponent = (1 << spec.exponent_bits) - 2
        else:
            max_exponent = (1 << spec.exponent_bits) - 1

        sign_bit = '0'
        exponent_bits = format(max_exponent, f'0{spec.exponent_bits}b')
        mantissa_bits = '1' * spec.mantissa_bits

        return sign_bit + exponent_bits + mantissa_bits

    def convert_between_formats(self, binary_string: str, from_format: FloatFormat, to_format: FloatFormat) -> str:
        """
        在不同格式之间转换

        参数:
            binary_string: 源格式的二进制字符串
            from_format: 源格式类型
            to_format: 目标格式类型

        返回:
            str: 目标格式的二进制字符串
        """
        # 先转换为浮点数
        float_value = self.binary_string_to_float(binary_string, from_format)

        # 再转换为目标格式
        return self.float_to_binary_string(float_value, to_format)

    def get_format_info(self, format_type: FloatFormat) -> Dict:
        """获取格式信息"""
        spec = self.format_specs[format_type]
        return {
            'name': spec.name,
            'total_bits': spec.total_bits,
            'sign_bits': spec.sign_bits,
            'exponent_bits': spec.exponent_bits,
            'mantissa_bits': spec.mantissa_bits,
            'exponent_bias': spec.exponent_bias,
            'has_implicit_bit': spec.has_implicit_bit,
            'supports_inf': spec.supports_inf,
            'supports_nan': spec.supports_nan,
            'is_unsigned': spec.is_unsigned,
            'is_exponent_only': spec.is_exponent_only,
            'is_mantissa_only': spec.is_mantissa_only,
        }

# 测试和示例代码
def test_converter():
    """测试转换器"""
    converter = UniversalFloatConverter()

    print("=== 通用浮点格式转换器测试 ===\n")

    # 测试用例
    test_cases = [
        (10.75, FloatFormat.FP32),
        (10.75, FloatFormat.FP16),
        (10.75, FloatFormat.BF16),
        (10.75, FloatFormat.TF32),
        (10.75, FloatFormat.FP64),
        (3.14, FloatFormat.FP8_E4M3),
        (3.14, FloatFormat.FP8_E5M2),
        (1.5, FloatFormat.FP6_E3M2),
        (1.5, FloatFormat.FP6_E2M3),
        (0.75, FloatFormat.FP4_E2M1),
    ]

    for value, format_type in test_cases:
        print(f"值: {value}")
        print(f"格式: {format_type.value}")

        # 转换为二进制
        binary_str = converter.float_to_binary_string(value, format_type)
        print(f"二进制: {binary_str}")

        # 转换回浮点数
        converted_back = converter.binary_string_to_float(binary_str, format_type)
        print(f"转换回: {converted_back}")

        # 格式信息
        info = converter.get_format_info(format_type)
        print(f"格式信息: {info}")
        print("-" * 50)

    # 测试特殊值
    print("\n=== 特殊值测试 ===")
    special_values = [float('inf'), float('-inf'), float('nan'), 0.0, -0.0]

    for value in special_values:
        print(f"值: {value}")
        for format_type in [FloatFormat.FP32, FloatFormat.FP16, FloatFormat.BF16, FloatFormat.FP8_E4M3]:
            try:
                binary_str = converter.float_to_binary_string(value, format_type)
                converted_back = converter.binary_string_to_float(binary_str, format_type)
                print(f"  {format_type.value}: {binary_str} -> {converted_back}")
            except Exception as e:
                print(f"  {format_type.value}: 错误 - {e}")
        print("-" * 30)

def test_converter_v2():
    """测试新的特殊格式"""
    converter = UniversalFloatConverter()

    print("=== FP8-E8M0 (UE8M0) 格式测试 ===")
    print("FP8-E8M0是无符号纯指数格式，只表示2的幂次方")
    print("动态范围: 2^-127 到 2^127")
    print("NaN用0xFF表示")
    print()

    # 测试FP8-E8M0
    e8m0_test_values = [1.0, 2.0, 4.0, 8.0, 16.0, 0.5, 0.25, 256.0, float('nan')]

    for val in e8m0_test_values:
        binary_str = converter.float_to_binary_string(val, FloatFormat.FP8_E8M0)
        converted_back = converter.binary_string_to_float(binary_str, FloatFormat.FP8_E8M0)
        print(f"值: {val}")
        print(f"二进制: {binary_str}")
        print(f"转换回: {converted_back}")
        if not math.isnan(val):
            print(f"是否2的幂: {abs(converted_back - val) < 1e-10}")
        print("-" * 30)

    print("\n=== FP4-E0M3 格式测试 ===")
    print("FP4-E0M3是有符号纯尾数格式，无指数位")
    print("本质上是INT4格式，使用查找表量化")
    print()

    # 测试FP4-E0M3
    e0m3_test_values = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -2.7, 3.14]

    for val in e0m3_test_values:
        try:
            binary_str = converter.float_to_binary_string(val, FloatFormat.FP4_E0M3)
            converted_back = converter.binary_string_to_float(binary_str, FloatFormat.FP4_E0M3)
            print(f"值: {val:6.2f} -> 二进制: {binary_str} -> 量化值: {converted_back:6.2f}")
        except Exception as e:
            print(f"值: {val:6.2f} -> 错误: {e}")

    print("\n=== 格式信息对比 ===")
    for fmt in [FloatFormat.FP8_E8M0, FloatFormat.FP4_E0M3]:
        info = converter.get_format_info(fmt)
        print(f"{fmt.value}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()

# Deprecated
def binary_to_number(
    s_s: str = "",
    s_e: str = "",
    s_m: str = "",
    p: int = 0,
    q: int = 0,
    r: int = 0,
    supports_inf: bool = True,
    supports_nan: bool = True,
):
    assert (len(s_s) == p)
    assert (len(s_e) == q)
    assert (len(s_m) == r)
    assert (set(s_s).issubset({'0', '1'}))
    assert (set(s_e).issubset({'0', '1'}))
    assert (set(s_m).issubset({'0', '1'}))

    b_IEEE754 = supports_inf and supports_nan
    set0 = {"0"}
    set1 = {"1"}
    set01 = {"0", "1"}
    # Special values
    # if b_IEEE754:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Inifnity
    #   4. Exponent all 1, Mantissa not all 0 -> NaN
    # elif supports_nan:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Normal value, not special
    #   4. Exponent all 1, Mantissa not all 0 -> Normal value, not special
    #   5. Exponent all 1, Mantissa all 1     -> NaN
    # elif supports_inf:
    #   No such case
    # else:
    #   1. Exponent all 0, Mantissa all 0     -> Zero
    #   2. Exponent all 0, Mantissa not all 0 -> Subnormal value, not special
    #   3. Exponent all 1, Mantissa all 0     -> Normal value, not special
    #   4. Exponent all 1, Mantissa not all 0 -> Normal value, not special
    #   5. Exponent all 1, Mantissa all 1     -> Normal value, not special
    if set(s_e) == set0 and set(s_m) == set0:
        return 0, 0, 0
    elif b_IEEE754 and set(s_e) == set1 and set(s_m) == set0:
        return float("inf"), "", ""
    elif (b_IEEE754 and set(s_e) == set1 and set(s_m) == set01) or (supports_nan and set(s_e) == set1 and set(s_m) == set1):
        return float("nan"), "", ""
    elif not b_IEEE754 and supports_inf:
        print("No such case")
        raise Exception

    i_s = int(s_s, 2) if len(s_s) > 0 else 0
    i_e = int(s_e, 2) if len(s_e) > 0 else 0
    i_m = int(s_m, 2) if len(s_m) > 0 else 0
    if "1" in s_e:  # Normal value
        e = i_e - (2 ** (q - 1) - 1)
        m = i_m * 2 ** (-r)
        value = (-1) ** i_s * 2 ** e * (m + 1)
    else:  # Subormal value
        e = 2 - 2 ** (q - 1)
        m = i_m * 2 ** (-r)
        value = (-1) ** i_s * 2 ** e * m
    a, b = f"{value:6.4e}".split("e")
    return value, a, b

def rgb(red_string, green_string, blue_string):
    return r"$\color{#D62728}{%s}\color{#2CA02C}{%s}\color{#1F77B4}{%s}$" % (red_string, green_string, blue_string)

def case_q0(ss: str = "", p: int = 0, r: int = 0, supports_inf: bool = True, supports_nan: bool = True):
    q = 0
    # Position Zero
    s_s, s_e, s_m = "0", "", "0" * r
    ss += "| %s | $+0$               | Positive Zero    |\n" % (rgb(s_s, s_e, s_m))

    # 1
    s_s, s_e, s_m = "0", "", "0" * (r - 1) + "1"
    ss += "| %s | $1$                |                  |\n" % (rgb(s_s, s_e, s_m))

    # Maximum
    s_s, s_e, s_m = "0", "", "1" * r
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, supports_inf, supports_nan)
    ss += "| %s | $%s\\times10^{%s}$ | Maximum |\n" % (rgb(s_s, s_e, s_m), a, b)

    # Negative Maximum
    s_s, s_e, s_m = "1", "", "1" * r
    _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, supports_inf, supports_nan)
    ss += "| %s | $%s\\times10^{%s}$ | Maximum negative |\n" % (rgb(s_s, s_e, s_m), a, b)

    # Negative Zero
    s_s, s_e, s_m = "1", "", "0" * r
    ss += "| %s | $-0$               | Positive Zero    |\n" % (rgb(s_s, s_e, s_m))

    # Print all non-negative values
    if p + 0 + r in [4, 6, 8]:
        ss += "\n+ All non-negative values\n\n"
        ss += "| Number(%s) | value |\n" % (rgb('Sign', 'Exponen', 'Mantissa'))
        ss += "| :-:        | :-:   |\n"

        for n in range(2 ** (p + q + r - 1)):
            binary = bin(n)[2:].zfill(p + q + r)
            s_s, s_e, s_m = binary[:p], binary[p:(p + q)], binary[(p + q):]
            value, _, _ = binary_to_number(s_s, s_e, s_m, p, q, r, supports_inf, supports_nan)
            #print(binary, s_s, s_e, s_m, value)
            ss += "| %s     | %.3e  |\n" % (rgb(s_s, s_e, s_m), value)
    return ss

def case_r0(ss: str = "", p: int = 0, q: int = 0, r: int = 0, supports_inf: bool = True, supports_nan: bool = True):
    r = 0
    # Position Zero
    ss += "| None         | $0$               | Zero (not exist)       |\n"

    # Maximum
    s_s, s_e, s_m = "" * p, "1" * q, ""
    a, b = f"{2 ** (2 ** (q-1)):6.4e}".split("e")
    ss += "| %s     | $%s\\times10^{%s}$  | Maximum  |\n" % (rgb(s_s, s_e, s_m), a, b)

    # Minimum
    s_s, s_e, s_m = "" * p, "0" * q, ""
    a, b = f"{-2 ** (2 ** (q-1)):6.4e}".split("e")
    ss += "| %s     | $%s\\times10^{%s}$ | Minimum |\n" % (rgb(s_s, s_e, s_m), a, b)

    # Alternative NaN
    s_s, s_e, s_m = "" * p, "1" * q, ""
    ss += "| %s         | $NaN$              | NaN     |\n" % (rgb(s_s, s_e, s_m))

    # Normal value less than 1/3 can both be represented when q >= 3
    if q >= 3:
        s_s, s_e, s_m = "" * p, "0" + "1" * (q - 3) + "01", "01" * (r // 2) + ("1" if r % 2 == 1 else "")
        _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, supports_inf, supports_nan)
        ss += "| %s     | $%s\\times10^{%s}$ | $\\frac{1}{3}$      |\n" % (rgb(s_s, s_e, s_m), a, b)

    # Print all non-negative values for FP4, FP6, FP8
    if p + q + r in [4, 6, 8]:
        ss += "\n+ All non-negative values\n\n"
        ss += "| Number(%s) | value |\n" % (rgb('Sign', 'Exponen', 'Mantissa'))
        ss += "| :-:        | :-:   |\n"

        for n in range(2 ** (p + q + r)):
            binary = bin(n)[2:].zfill(p + q + r)
            s_s, s_e, s_m = binary[:p], binary[p:(p + q)], binary[(p + q):]
            value = 2 ** (int(s_e, 2) - (2 ** (q - 1) - 1))
            if n == 2 ** (p + q + r) - 1:
                value = float("nan")
            #print(binary, s_s, s_e, s_m, value)
            ss += "| %s     | %.3e  |\n" % (rgb(s_s, s_e, s_m), value)

    return ss

def case_general(ss: str = "", float_format: FloatFormat = None):

    spec = spec = FORMAT_SPECS[float_format]
    p, q, r = spec.sign_bits, spec.exponent_bits, spec.mantissa_bits
    supports_inf = spec.supports_inf
    supports_nan = spec.supports_nan
    has_implicit_bit = spec.has_implicit_bit
    b_IEEE754 = supports_inf and supports_nan and has_implicit_bit

    def integer_to_binary(value: int = 0, width: int = 1):
        return bin(value)[2:].zfill(width)

    def sem10_to_bits_tuple(x: list):
        return integer_to_binary(x[0], spec.sign_bits), integer_to_binary(x[1], spec.exponent_bits), integer_to_binary(x[2], spec.mantissa_bits)

    def sem10_to_bits(x: list):
        return "".join(sem10_to_bits_tuple(x))

    def sem10_to_rgb_bits(x: list):
        return rgb(*sem10_to_bits_tuple(x))

    converter = UniversalFloatConverter()

    def get_format_dict(sem10: list, comment: str, special_value_string: str = ""):
        if special_value_string == "":
            m, e = f"{converter.binary_string_to_float(sem10_to_bits(sem10), FloatFormat[spec.name]):8.6e}".split("e")
        return {
            "rgb_string": sem10_to_rgb_bits(sem10),
            "value_string": f"${m}\\times10^{{{e}}}$" if special_value_string == "" else special_value_string,
            "comment": comment,
        }

    ss_template = r"| {rgb_string} | {value_string} | {comment} |" + "\n"

    # Position Zero
    ss += ss_template.format(**get_format_dict([0, 0, 0], "Positive Zero", "$+0$"))

    # Smallest subnormal value
    ss += ss_template.format(**get_format_dict([0, 0, 1], "Minimum Subnormal"))

    # Largest subnormal value
    ss += ss_template.format(**get_format_dict([0, 0, 2 ** r - 1], "Maximum Subnormal"))

    # Smallest normal value
    if q >= 1:
        ss += ss_template.format(**get_format_dict([0, 1, 0], "Minimum Normal"))
    else:
        # FP4E0M3
        raise Exception("No support for q == 0")

    # Largest number < 1
    if q == 1:  # For FP4-E1M2
        s_s, s_e, s_m = "0", "0", "1" * r
        ss += "| %s     | $1 - 2^{%d}$       | Largest number < 1  |\n" % (rgb(s_s, s_e, s_m), -(r + 1))
        raise Exception("Unchecked")
    else:
        s_s, s_e, s_m = "0", "0" + "1" * (q - 2) + "0", "1" * r
        if q == 2 and r == 3:  # special case for FP6-E2M3 since smallest normal value is 1
            ss += "| %s     | $1 - 2^{%d}$       | Largest number < 1  |\n" % (rgb(s_s, s_e, s_m), -r)
            raise Exception("Unchecked")
        else:
            ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 2, 2 ** r - 1], "Largest number < 1", f"$1 - 2 ^ {{-{r+1}}}$"))

    # 1
    ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 1, 0], "", "$1$"))

    # Smallest number > 1
    ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 1, 1], "Smallest number > 1", f"$1 + 2 ^ {{-{r}}}$"))

    # 2
    ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1), 0], "", "$2$"))

    # 3
    ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1), 2 ** (r - 1)], "", "$3$"))

    # 4
    if q >= 3:
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) + 1, 0], "", "$4$"))
    else:
        # 4 can only be represented when q >= 3
        raise Exception("No support for q <= 2")

    # 5 and 6 can only be represented when q >= 3 and r >= 2
    # 5
    if q >= 3 and r >= 2:
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) + 1, 2 ** (r - 2)], "", "$5$"))
    # 6
    if q >= 3 and r >= 2:
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) + 1, 2 ** (r - 1)], "", "$6$"))

    # Maximum
    if b_IEEE754:
        s_s, s_e, s_m = "0", "1" * (q - 1) + "0", "1" * r
        ss += ss_template.format(**get_format_dict([0, 2 ** q - 2, 2 ** r - 1], "Positive maximum"))
    else:
        ss += ss_template.format(**get_format_dict([0, 2 ** q - 2, 2 ** r - 1], "Positive maximum"))  # FP8-E4M3 checked
        if supports_nan:  # exponent all 1 is normal value, and mantissa can not be all 1
            # s_s, s_e, s_m = "0", "1" * q, "1" * (r - 1) + "0"
            pass
        else:  # exponent all 1 is normal value, and mantissa can be all 1
            # s_s, s_e, s_m = "0", "1" * q, "1" * r
            pass
    # _, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, supports_inf, supports_nan)
    # ss += "| %s     | $%s\\times10^{%s}$ | Maximum |\n" % (rgb(s_s, s_e, s_m), a, b)

    # Negative Maximum
    if b_IEEE754:
        s_s, s_e, s_m = "1", "1" * (q - 1) + "0", "1" * r
        ss += ss_template.format(**get_format_dict([1, 2 ** q - 2, 2 ** r - 1], "Negative maximum"))
    else:
        ss += ss_template.format(**get_format_dict([1, 2 ** q - 2, 2 ** r - 1], "Negative maximum"))  # FP8-E4M3 checked
        if supports_nan:  # exponent all 1 is normal value, and mantissa can not be all 1
            s_s, s_e, s_m = "1", "1" * q, "1" * (r - 1) + "0"
        else:  # exponent all 1 is normal value, and mantissa can be all 1
            s_s, s_e, s_m = "1", "1" * q, "1" * r
    #_, a, b = binary_to_number(s_s, s_e, s_m, p, q, r, supports_inf, supports_nan)
    #ss += "| %s     | $%s\\times10^{%s}$ | Maximum negative    |\n" % (rgb(s_s, s_e, s_m), a, b)

    # Negative Zero
    ss += ss_template.format(**get_format_dict([1, 0, 0], "Negative Zero", "$-0$"))

    if supports_inf:
        # Positive Infinity
        ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 0], "Positive Infinity", r"$+\infty$"))

        # Negative Infinity
        ss += ss_template.format(**get_format_dict([1, 2 ** q - 1, 0], "Negative Infinity", r"$-\infty$"))

    if b_IEEE754:
        # Signalling Not a Number
        ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 1], "Signalling NaN", "sNaN"))

        # another kind of NaN can both be represented when r >= 2
        if r >= 2:
            # Quiet Not a Number
            ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 2 ** (r - 1) + 1], "Quiet NaN", "qNaN"))

    if supports_nan:
        # Alternative NaN
        ss += ss_template.format(**get_format_dict([0, 2 ** q - 1, 2 ** r - 1], "NaN", "NaN"))

    # Normal value less than 1/3 can both be represented when q >= 3
    if q >= 3:
        rr = int((2 ** r - 1) / 3) if r % 2 == 0 else int((2 ** r + 1) / 3)
        ss += ss_template.format(**get_format_dict([0, 2 ** (q - 1) - 3, rr], "$\\frac{1}{3}$"))

    # Print all values for FP4, FP6, FP8
    # TODO: chagne to 2D table
    if p + q + r in [4, 6, 8]:
        ss += "\n+ All non-negative values\n\n"
        ss += "| Number(%s) | value |\n" % (rgb('Sign', 'Exponen', 'Mantissa'))
        ss += "| :-:        | :-:   |\n"

        for n in range(2 ** (p + q + r - 1)):
            binary = bin(n)[2:].zfill(p + q + r)
            s_s, s_e, s_m = binary[:p], binary[p:(p + q)], binary[(p + q):]
            value, _, _ = binary_to_number(s_s, s_e, s_m, p, q, r, supports_inf, supports_nan)
            #print(binary, s_s, s_e, s_m, value)
            ss += "| %s     | %.3e  |\n" % (rgb(s_s, s_e, s_m), value)

    return ss

# Generate Latex in Markdown:
# For variables for evaluation but not need "{}" in Latex, use "{vv}"
# For variables not for evaluation but need "{}" in Latex, use "{{vv}}"
# For variables both for evaluation and need "{}" in Latex, use "{{{vv}}}"

normal_value_template = r"""+ Normal value (${normal_exponent_min}_2 \le e_2 \le {normal_exponent_max}_2$) (2 in subscript represents the base)

$$
\begin{{equation}}
\begin{{aligned}}
E &= e - \left( 2 ^ {{q-1}} - 1 \right) = e - {normal_bias} \\
M &= m \cdot 2 ^ {{-r}} = m \cdot 2 ^ {{-{r}}} \\
\text{{value}} &= \left( -1 \right) ^ {{s}} 2 ^ {{E}} \left( 1 + M \right) = \left( -1 \right) ^ {{s}} 2 ^ {{{neg_r_minus_normal_bias}}} 2 ^ {{e}} \left( m + 2 ^ {{{r}}} \right)
\end{{aligned}}
\end{{equation}}
$$

"""

subnormal_value_template = r"""+ Subnormal value ($e_2 = {subnormal_exponent_zeros}_2$)

$$
\begin{{equation}}
\begin{{aligned}}
E &= 2 - 2 ^ {{q-1}} = -{subnormal_bias} \\
M &= m \cdot 2 ^ {{-r}} = m \cdot 2 ^ {{-{r}}} \\
\text{{value}} &= \left( -1 \right) ^ {{s}} 2 ^ {{E}} M = \left( -1 \right) ^ {{s}} 2 ^ {{{neg_r_minus_subnormal_bias}}} m
\end{{aligned}}
\end{{equation}}
$$

"""

def build_md(float_format: FloatFormat):

    spec = FORMAT_SPECS[float_format]
    p, q, r = spec.sign_bits, spec.exponent_bits, spec.mantissa_bits
    supports_inf = spec.supports_inf
    supports_nan = spec.supports_nan
    has_implicit_bit = spec.has_implicit_bit
    b_IEEE754 = supports_inf and supports_nan and has_implicit_bit

    comment = ""
    if spec.name == "FP4-E2M1":
        comment = " (MXFP4 or NVFP4)"

    ss = ""
    ss += f"# {spec.name}{comment} - {'' if b_IEEE754 else 'not '}IEEE754\n"
    ss += "\n"
    ss += f"+ Sign:     $s$ ($p = {p}$ bit)\n"
    ss += f"+ Exponent: $e$ ($q = {q}$ bit)\n"
    ss += f"+ Mantissa: $m$ ($r = {r}$ bit)\n"
    ss += "\n"

    ss += normal_value_template.format(
        normal_exponent_min=bin(1)[2:].zfill(q),
        normal_exponent_max=bin(2 ** q - 2)[2:].zfill(q),
        normal_bias=2 ** (q - 1) - 1,
        r=r,
        neg_r_minus_normal_bias=-r - 2 ** (q - 1) + 1,
    )

    ss += subnormal_value_template.format(
        subnormal_exponent_zeros=bin(0)[2:].zfill(q),
        subnormal_bias=2 ** (q - 1) - 2,
        r=r,
        neg_r_minus_subnormal_bias=-r - 2 ** (q - 1) + 2,
    )

    ss += "+ Special value\n"
    if not supports_inf:
        ss += "  - **No bits pattern for infinity.**\n"
    if not supports_nan:
        ss += "  - **No bits pattern for NaN.**\n"
    ss += "\n"
    ss += "| Exponent  | Mantissa  | meaning         |\n"
    ss += "| :-:       | :-:       | :-:             |\n"
    ss += "| all 0     | all 0     | Signed Zero     |\n"
    ss += "| all 0     | not all 0 | Subnormal Value |\n"
    if supports_inf:
        ss += "| all 1 | all 0     | Signed Infinity |\n"
    else:
        ss += "| all 1 | all 0     | Normal value    |\n"
    if supports_nan:
        ss += "| all 1 | not all 0 | NaN      |\n"
    else:
        ss += "| all 1 | not all 0 | Normal value    |\n"
    ss += "\n"

    ss += "+ Examples\n\n"
    ss += f"| Number({rgb('Sign', 'Exponent', 'Mantissa')}) | value | comment |\n"
    ss += "|:-:|:-:|:-:|\n"

    if q == 0:
        ss = case_q0(ss, p, r, supports_inf, supports_nan)
    elif r == 0:
        ss = case_r0(ss, p, q, supports_inf, supports_nan)
    else:
        ss = case_general(ss, float_format)

    with open(f"output/{spec.name}.md", "w") as f:
        f.write(ss)

    return

if __name__ == "__main__":
    # test_converter()

    # test_converter_v2()

    # for float_name, float_format in FloatFormat.__members__.items():
    for float_name, float_format in [  #("FP32", FloatFormat.FP32),
        #("FP16", FloatFormat.FP16),
        #("TF32", FloatFormat.TF32),
        #("BF16", FloatFormat.BF16),
        #("FP64", FloatFormat.FP64),
        ("FP8-E4M3", FloatFormat.FP8_E4M3),
        #("FP8-E5M2", FloatFormat.FP8_E5M2),
        #("FP8-E8M0", FloatFormat.FP8_E8M0),
        #("FP6-E2M3", FloatFormat.FP6_E2M3),
        #("FP6-E3M2", FloatFormat.FP6_E3M2),
        #("FP4-E2M1", FloatFormat.FP4_E2M1),
        #("FP4-E0M3", FloatFormat.FP4_E0M3),
    ]:
        print(f"Build {float_name}")
        build_md(float_format)

    print("Finish")
"""
每个浮点数的四种表示方法：
十进制数值：    value=314.159265
sem2 表示：    s=0_2, e=10000111_2, m=00111010001010001100011_2
sem10表示：    s=0_10, e=135_10, m=1905763_10
科学计数表示:   x=3.14159265, y=2
"""
