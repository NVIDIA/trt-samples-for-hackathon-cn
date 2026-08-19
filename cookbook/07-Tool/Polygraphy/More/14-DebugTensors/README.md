# Debug tensors

+ Read an intermediate tensor without turning it into an engine output.

+ Steps to run.

```bash
python3 main.py
```

`MarkDebug` marks tensors in the network; at run time a `trt.IDebugListener` is
called with a device pointer each time one is written. The selling point over the
obvious alternative — promoting the tensor to an output — is that the engine's
I/O signature does not change:

```
engine.is_debug_tensor('relu'): True
engine I/O tensors : ['x', 'y', 'z']
inference returned : ['y', 'z']
captured 'relu': shape (1, 32, 28, 28), range [0.0000, 272.0352]
```

The buffer handed to the callback is only valid for its duration, so anything
worth keeping has to be copied out inside the callback.

## The call that looks required is a no-op

The obvious reading is that `set_tensor_debug_state(name, True)` arms a marked
tensor. It does not — `MarkDebug` already leaves the state on:

```
listener, state untouched : debug state was already True, captured 1 tensor(s)
listener, state set True  : debug state was already True, captured 1 tensor(s)
listener, state set False : debug state was already True, captured 0 tensor(s)
no listener               : debug state was already True, captured 0 tensor(s)
```

Only two things change the outcome: setting the state to `False`, and forgetting
the listener. The second fails the quiet way — the inference runs, the outputs are
correct, and an empty dict reads like "the tensor was never written" rather than
"you forgot a call".

`False` is the direction worth using: mute one tensor for one run, no rebuild, on
an engine where every candidate was marked at build time.

## It is not free, and the bill is at build time

None of these runs attaches a listener or enables anything — the debug tensor is
never read:

```
baseline          :  0.605 ms, workspace  125440 B
MarkDebug         :  0.943 ms, workspace  201216 B
mark_output       :  0.740 ms, workspace   75264 B
```

`MarkDebug` costs **1.56x** over baseline with the debug state never switched on,
because a tensor that must stay observable cannot be fused away. So this is a
debugging build, not a flag to leave on.

Against the alternative from
[`../13-PerLayerPrecision/`](../13-PerLayerPrecision/README.md):

```
MarkDebug     : I/O = ['x', 'y', 'z']
mark_output   : I/O = ['x', 'y', 'z', 'relu']
```

`MarkDebug` keeps the contract; `mark_output` changes it for every caller, every
saved plan and every downstream tool. But `mark_output` needs no listener, no
callback per inference, and is the cheaper of the two here (1.22x).

## Every unfused tensor

`NetworkFromOnnxPath(mark_unfused_tensors_as_debug_tensors=True)` marks whatever
survived fusion. Which tensors those are is not known until the build finishes, so
nothing shows on the `INetworkDefinition` — `network.is_debug_tensor` reports zero
either way, and looking there is what makes the kwarg seem to do nothing. The
effect is only visible at run time:

```
mark_unfused_tensors_as_debug_tensors=True : captured 9 tensor(s)
  names: ['__myln_k_arg__bb1_3_myl4', '__myln_k_arg__bb1_5_myl4', ...]
mark_unfused_tensors_as_debug_tensors=False: captured 0 tensor(s)
```

The names that come back are TensorRT's post-fusion internal ones, so this shows
*that* a value went wrong, not which layer of the original model produced it.
