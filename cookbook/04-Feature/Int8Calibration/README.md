# Int8 Calibration

+ Post-training INT8 quantization with a custom Python calibrator.

+ Covers the four `trt.CalibrationAlgoType` members and their matching calibrator base classes
  (`trt.IInt8Calibrator`, `trt.IInt8EntropyCalibrator`, `trt.IInt8EntropyCalibrator2`,
  `trt.IInt8MinMaxCalibrator`, `trt.IInt8LegacyCalibrator`), `get_algorithm`, and
  `IRuntimeConfig.get_execution_context_allocation_strategy`.

+ Steps to run.

```bash
python3 main.py
```
