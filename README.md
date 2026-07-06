# onnxpascal — ONNX Runtime inference for Object Pascal

[![ci](https://github.com/ebalvis/onnxpascal/actions/workflows/ci.yml/badge.svg)](https://github.com/ebalvis/onnxpascal/actions/workflows/ci.yml)

Run trained ML models (`.onnx`) directly from **Free Pascal / Delphi** by calling
the [ONNX Runtime](https://onnxruntime.ai) C API **natively — no C toolchain, no
CGO**. Object Pascal binds native shared libraries out of the box, so this is a
lightweight bridge to deploy models trained in Python into Pascal desktop and
industrial/edge software.

- **Zero build-time native deps:** a single `.pas` unit; only `SysUtils`. No wrapper DLL, no C compiler.
- **Runtime dep:** the ONNX Runtime shared library (`onnxruntime.dll` / `libonnxruntime.so`), which `pip install onnxruntime` provides.
- **Tensor types:** `float32`, `float64`, `int32`, `int64`, `uint8`; multiple inputs/outputs; model introspection.
- **Fail-early shape validation:** a shape that does not match the data raises `EOnnx` *before* calling ORT (no out-of-bounds read).
- **Deterministic warm path:** `TOnnxWarmRunner` reuses the input tensor and names across calls — lower latency and far less jitter.
- **Conformal uncertainty (`uOnnxConformal`):** split-conformal prediction intervals (regression) and prediction sets (classification) with guaranteed 1−α coverage.
- **Verified:** FPC 3.2 on **Windows** and **Linux** (Debian) and **Delphi** (RAD Studio 37), against ONNX Runtime 1.27 (API v28); **12/12 tests** on all three.

## Quick start

```pascal
uses uOnnxRuntime;
var s: TOnnxSession; x: TSingleArray; shp: TInt64Array; y: TSingleArray;
begin
  s := TOnnxSession.Create('model.onnx');
  try
    SetLength(x, 3);   x[0]:=1; x[1]:=2; x[2]:=3;
    SetLength(shp, 2); shp[0]:=1; shp[1]:=3;
    y := s.Run(s.InputNames[0], s.OutputNames[0], x, shp);   // -> [1.5]
  finally
    s.Free;
  end;
end;
```

## Build & run (Linux, todo en uno)

```bash
pip install onnxruntime      # aporta libonnxruntime.so
./build.sh                   # localiza la .so, compila y ejecuta test (12/12) + bench
```

## Build & run the example/test (manual)

```bash
# 1) test model (linear regression 3->1):  pip install scikit-learn skl2onnx onnx
python tools/make_test_model.py

# 2) compile (FPC; no gcc needed)
fpc -FEbin -FUlib -Fusrc examples/predict.pas
fpc -FEbin -FUlib -Fusrc tests/test_onnx.pas

# 3) put the native lib next to the exe (from `pip install onnxruntime`)
cp .../site-packages/onnxruntime/capi/onnxruntime.dll bin/

# 4) run
cd bin && ./test_onnx.exe        # PASS: [1,2,3] -> 1.5000
./predict.exe                    # inputs/outputs + Run
```

## Benchmark de latencia (cold vs warm)

```bash
fpc -FEbin -FUlib -Fusrc bench/bench.pas
cd bin && ./bench.exe model.onnx 20000            # cold + warm
./bench.exe model.onnx 20000 5.0                  # + contrato: falla si p99 warm > 5.0 µs
```
Compara la ruta **cold** (`Run` crea tensor + nombres por llamada) con la **warm**
(`TOnnxWarmRunner`, reutiliza el `OrtValue` de entrada). Reporta min/mean/p50/p95/p99/max
y jitter. Con el modelo de prueba (regresión 3→1) en CPU, la ruta warm baja la mediana
a **~1.7–2.5 µs** y **reduce el jitter en un orden de magnitud**. Un tercer argumento
(µs) activa una **puerta p99** que sale con código 1 si se supera — útil como gate de CI.

## How it works
The `OrtApi` is a table of function pointers returned by `OrtGetApiBase()->GetApi()`.
The unit accesses each function **by index** (taken from the official
`onnxruntime_c_api.h`), which avoids replicating the full struct layout — a single
misplaced field would shift every offset. Only `OrtGetApiBase` is imported by name.

## Multiple inputs/outputs and int64

```pascal
// N entradas (float32/int64) -> M salidas, cada una con su tipo, forma y datos:
SetLength(ins, 1); ins[0] := OnnxInt64('x', xi, shp);      // OnnxFloat(...) para float32
SetLength(names, 2); names[0] := 'out_a'; names[1] := 'out_b';
outs := sess.RunMulti(ins, names);                          // outs[j].ElemType, .Shape, .DataF/.DataI

// Multi-ENTRADA (2 entradas float32 -> 1 salida 'y'):
SetLength(ins, 2); ins[0] := OnnxFloat('a', a, shp); ins[1] := OnnxFloat('b', b, shp);
SetLength(names, 1); names[0] := 'y';
outs := sess.RunMulti(ins, names);                          // outs[0].DataF
```

## Warm path (low jitter)

```pascal
uses uOnnxRuntime;
var w: TOnnxWarmRunner; buf, y: TSingleArray; shp: TInt64Array;
begin
  SetLength(shp, 2); shp[0]:=1; shp[1]:=3;
  w := TOnnxWarmRunner.Create(s, s.InputNames[0], s.OutputNames[0], shp);
  try
    buf := w.InputBuffer;                 // escribe aquí (no cambies el tamaño)
    buf[0]:=1; buf[1]:=2; buf[2]:=3;
    y := w.Infer;                          // reutiliza el OrtValue de entrada -> menos jitter
  finally w.Free; end;
end;
```

## Conformal uncertainty

```pascal
uses uOnnxConformal;
// Regresión: intervalo con cobertura 1-alpha a partir de residuos held-out |y-ŷ|.
reg := TConformalRegressor.Create(calibResiduals, 0.10);   // 90%
reg.Interval(yhat, lo, hi);                                 // [lo, hi]

// Clasificación (LAC): prediction set a partir de prob(clase verdadera) en calibración.
clf := TConformalClassifier.Create(calibProbTrue, 0.10);
setIdx := clf.PredictionSet(probs);                        // índices de clase incluidos
```
Ver el ejemplo completo en [`examples/predict_conformal.pas`](examples/predict_conformal.pas).

## Scope (v0.3)
Múltiples entradas/salidas; tensores **float32/float64/int32/int64/uint8**; **validación de
shape** (fail-early); **ruta warm** de baja latencia/jitter con contrato p99 para CI; y una
capa de **incertidumbre conforme** (intervalos de regresión + prediction sets). **No** incluido
por diseño: *execution providers* de GPU y entrenamiento (ver más abajo).

## Related work / positioning
[TONNXRuntime](https://github.com/hshatti/TONNXRuntime) (MIT) is a mature, full ONNX Runtime
binding for Free Pascal/Delphi: header translation, generics for all tensor types, GPU
execution providers (CUDA/TensorRT/OpenVINO/DirectML), IoBinding and on-device training.
**If you need GPU or the full feature surface, use it.**

`onnxpascal` deliberately targets a different niche — **minimal, auditable, edge/industrial**:
- **Index-based binding** (no header translation): a single small unit you can audit in one sitting.
- **Fail-early shape validation** (avoids silent out-of-bounds reads).
- **Deterministic warm path** with a p99 latency contract for CI.
- **Conformal uncertainty** built in (guaranteed coverage) — the piece an inspection / soft-sensing pipeline actually needs, and which general bindings do not provide.

## License
MIT — see [LICENSE](LICENSE).
