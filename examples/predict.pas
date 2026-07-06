program predict;
{ Ejemplo de uso de uOnnxRuntime: carga un modelo, muestra sus entradas/salidas
  y ejecuta una inferencia.  Uso:  predict [model.onnx] }
{$IFDEF FPC}{$mode objfpc}{$H+}{$ENDIF}
uses
  SysUtils, uOnnxRuntime;
var
  sess: TOnnxSession;
  modelPath, s, nm: string;
  x: TSingleArray;
  shp: TInt64Array;
  y: TSingleArray;
begin
  if ParamCount >= 1 then modelPath := ParamStr(1) else modelPath := 'model.onnx';
  WriteLn('onnxruntime ', OnnxRuntimeVersion);

  sess := TOnnxSession.Create(modelPath);
  try
    s := '';
    for nm in sess.InputNames do s := s + nm + ' ';
    WriteLn('inputs : ', s);
    s := '';
    for nm in sess.OutputNames do s := s + nm + ' ';
    WriteLn('outputs: ', s);

    SetLength(x, 3);   x[0] := 1; x[1] := 2; x[2] := 3;
    SetLength(shp, 2); shp[0] := 1; shp[1] := 3;
    y := sess.Run(sess.InputNames[0], sess.OutputNames[0], x, shp);
    WriteLn(Format('Run [1,2,3] -> %.4f', [y[0]]));
  finally
    sess.Free;
  end;
end.
