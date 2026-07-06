program predict_conformal;
{ Ejemplo end-to-end: inferencia ONNX + incertidumbre conforme.

  Patrón industrial "train-in-Python, serve-in-Pascal-at-the-edge":
  un modelo entrenado (aquí una regresión lineal exportada a model.onnx) da una
  predicción PUNTUAL; la capa conforme la envuelve en un INTERVALO con cobertura
  marginal garantizada 1-alpha, usando residuos de un conjunto de calibración
  (held-out) — sin reentrenar el modelo.

  Genera antes:  python tools/make_test_model.py
  Compila:       fpc -Mobjfpc -Fusrc -FE. examples/predict_conformal.pas }
{$IFDEF FPC}{$mode objfpc}{$H+}{$ENDIF}
uses
  SysUtils, uOnnxRuntime, uOnnxConformal;

var
  s: TOnnxSession;
  reg: TConformalRegressor;
  x, y: TSingleArray; shp: TInt64Array;
  calibResiduals: TDoubleArray;
  yhat, lo, hi: Double;
  alpha: Double;
  i: Integer;
begin
  // 1) Cargar el modelo y predecir un punto.
  s := TOnnxSession.Create('model.onnx');
  try
    SetLength(x, 3);   x[0] := 1; x[1] := 2; x[2] := 3;
    SetLength(shp, 2); shp[0] := 1; shp[1] := 3;
    y := s.Run(s.InputNames[0], s.OutputNames[0], x, shp);
    yhat := y[0];
  finally
    s.Free;
  end;

  // 2) Residuos de calibración |y - ŷ| de un conjunto held-out.
  //    (Aquí simulados; en producción se calculan sobre datos reales apartados).
  SetLength(calibResiduals, 15);
  for i := 0 to 14 do
    calibResiduals[i] := 0.05 + i * 0.03;   // 0.05, 0.08, ... 0.47

  // 3) Envolver la predicción con cobertura 1-alpha = 90%.
  alpha := 0.10;
  reg := TConformalRegressor.Create(calibResiduals, alpha);
  try
    reg.Interval(yhat, lo, hi);
    WriteLn(Format('onnxruntime %s', [OnnxRuntimeVersion]));
    WriteLn(Format('input          : [%.0f, %.0f, %.0f]', [x[0], x[1], x[2]]));
    WriteLn(Format('prediction yhat: %.4f', [yhat]));
    WriteLn(Format('coverage       : %.0f%%  (alpha=%.2f)', [(1 - alpha) * 100, alpha]));
    WriteLn(Format('half-width q   : %.4f', [reg.Q]));
    WriteLn(Format('interval 90%%   : [%.4f, %.4f]', [lo, hi]));
  finally
    reg.Free;
  end;
end.
