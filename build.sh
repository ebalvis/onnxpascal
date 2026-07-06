#!/bin/bash
# build.sh — compila y prueba onnxpascal en Linux con Free Pascal.
#
# Requisitos:
#   - fpc (Free Pascal >= 3.2)
#   - libonnxruntime.so  (p.ej. de `pip install onnxruntime`)
#   - para generar los modelos de prueba: python3 con onnx, skl2onnx, scikit-learn
#     (si no, copia tus .onnx a ./bin antes de ejecutar)
#
# Uso:
#   ./build.sh
#   ONNXRT_SO=/ruta/libonnxruntime.so.1.27.0 ./build.sh    # forzar ruta de la .so
#   OUT=/otro/dir ./build.sh                                # otro directorio de salida
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$DIR/bin}"
LIB="$OUT/native"
mkdir -p "$OUT" "$LIB"

# 1) localizar libonnxruntime.so (ONNXRT_SO override, o via el paquete python onnxruntime)
SO="${ONNXRT_SO:-}"
if [ -z "$SO" ]; then
  SO=$(python3 -c "import onnxruntime,glob,os;d=os.path.dirname(onnxruntime.__file__);print((glob.glob(d+'/capi/libonnxruntime.so.*') or [''])[0])" 2>/dev/null || true)
fi
if [ -z "$SO" ] || [ ! -e "$SO" ]; then
  echo "ERROR: no encuentro libonnxruntime.so." >&2
  echo "  Instala onnxruntime (pip install onnxruntime) o exporta ONNXRT_SO=/ruta/libonnxruntime.so.X" >&2
  exit 1
fi
# nombre base (para el link -l) y SONAME (para el runtime)
ln -sf "$SO" "$LIB/libonnxruntime.so"
ln -sf "$SO" "$LIB/libonnxruntime.so.1"
echo "libonnxruntime: $SO"

# 2) modelos de prueba (generar si faltan y hay libs; si no, deben estar ya en $OUT)
if [ ! -f "$OUT/model.onnx" ]; then
  if python3 -c "import onnx, skl2onnx" 2>/dev/null; then
    ( cd "$OUT" && python3 "$DIR/tools/make_test_model.py" \
                && python3 "$DIR/tools/make_extra_models.py" \
                && python3 "$DIR/tools/make_types_models.py" )
  else
    echo "AVISO: faltan modelos y no hay onnx/skl2onnx para generarlos." >&2
    echo "  Genera model*.onnx (tools/*.py) o cópialos a $OUT" >&2
  fi
fi

# 3) compilar (-Fl añade la ruta de la .so para el enlace)
echo "=== compilando ==="
fpc -Fu"$DIR/src" -Fl"$LIB" -FE"$OUT" -FU"$OUT" "$DIR/tests/test_onnx.pas"
fpc -Fu"$DIR/src" -Fl"$LIB" -FE"$OUT" -FU"$OUT" "$DIR/bench/bench.pas"

# 4) ejecutar
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"
echo "=== test ==="
( cd "$OUT" && ./test_onnx )
echo "=== bench ==="
( cd "$OUT" && ./bench model.onnx 20000 )
