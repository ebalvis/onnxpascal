program test_onnx;
{ Tests de humo: float single (linear -> 1.5), int64 (x+10), y multi-salida (id, x2).
  Genera antes:  python tools/make_test_model.py ; python tools/make_extra_models.py
  Sale con código 0 (todo OK) o 1 (algún fallo). }
{$IFDEF FPC}{$mode objfpc}{$H+}{$ENDIF}
uses
  SysUtils, uOnnxRuntime, uOnnxConformal;

var
  failures: Integer = 0;

procedure Pass(const m: string); begin WriteLn('PASS: ', m); end;
procedure Fail(const m: string); begin WriteLn('FAIL: ', m); Inc(failures); end;

procedure TestLinearFloat;
var s: TOnnxSession; x, y: TSingleArray; shp: TInt64Array;
begin
  s := TOnnxSession.Create('model.onnx');
  try
    SetLength(x, 3);   x[0]:=1; x[1]:=2; x[2]:=3;
    SetLength(shp, 2); shp[0]:=1; shp[1]:=3;
    y := s.Run(s.InputNames[0], s.OutputNames[0], x, shp);
  finally s.Free; end;
  if (Length(y) = 1) and (Abs(y[0] - 1.5) < 0.01) then
    Pass(Format('float single: linear [1,2,3] -> %.3f', [y[0]]))
  else
    Fail(Format('float single: esperaba 1.5, dio %.3f', [y[0]]));
end;

procedure TestInt64;
var s: TOnnxSession; xi: TInt64Array; shp: TInt64Array; ins: TOnnxTensorArray;
    names: TStringArray; outs: TOnnxTensorArray;
begin
  s := TOnnxSession.Create('model_int64.onnx');
  try
    SetLength(xi, 3);  xi[0]:=1; xi[1]:=2; xi[2]:=3;
    SetLength(shp, 1); shp[0]:=3;
    SetLength(ins, 1); ins[0] := OnnxInt64('x', xi, shp);
    SetLength(names, 1); names[0] := 'y';
    outs := s.RunMulti(ins, names);
  finally s.Free; end;
  if (Length(outs) = 1) and (outs[0].ElemType = oeInt64) and (Length(outs[0].DataI) = 3)
     and (outs[0].DataI[0] = 11) and (outs[0].DataI[1] = 12) and (outs[0].DataI[2] = 13) then
    Pass('int64: [1,2,3] + 10 -> [11,12,13]')
  else
    Fail('int64: no dio [11,12,13]');
end;

procedure TestMultiOutput;
var s: TOnnxSession; x: TSingleArray; shp: TInt64Array; ins: TOnnxTensorArray;
    names: TStringArray; outs: TOnnxTensorArray;
begin
  s := TOnnxSession.Create('model_multi.onnx');
  try
    SetLength(x, 3);   x[0]:=1; x[1]:=2; x[2]:=3;
    SetLength(shp, 1); shp[0]:=3;
    SetLength(ins, 1); ins[0] := OnnxFloat('x', x, shp);
    SetLength(names, 2); names[0] := 'out_id'; names[1] := 'out_dbl';
    outs := s.RunMulti(ins, names);
  finally s.Free; end;
  if (Length(outs) = 2)
     and (Length(outs[0].DataF) = 3) and (Abs(outs[0].DataF[0]-1) < 0.01) and (Abs(outs[0].DataF[2]-3) < 0.01)
     and (Length(outs[1].DataF) = 3) and (Abs(outs[1].DataF[0]-2) < 0.01) and (Abs(outs[1].DataF[2]-6) < 0.01) then
    Pass('multi-salida: out_id=[1,2,3], out_dbl=[2,4,6]')
  else
    Fail('multi-salida: valores incorrectos');
end;

procedure TestMultiInput;
var s: TOnnxSession; a, b: TSingleArray; shp: TInt64Array; ins: TOnnxTensorArray;
    names: TStringArray; outs: TOnnxTensorArray;
begin
  s := TOnnxSession.Create('model_add.onnx');
  try
    SetLength(a, 3); a[0]:=1;  a[1]:=2;  a[2]:=3;
    SetLength(b, 3); b[0]:=10; b[1]:=20; b[2]:=30;
    SetLength(shp, 1); shp[0]:=3;
    SetLength(ins, 2); ins[0] := OnnxFloat('a', a, shp); ins[1] := OnnxFloat('b', b, shp);
    SetLength(names, 1); names[0] := 'y';
    outs := s.RunMulti(ins, names);
  finally s.Free; end;
  if (Length(outs) = 1) and (Length(outs[0].DataF) = 3)
     and (Abs(outs[0].DataF[0]-11) < 0.01) and (Abs(outs[0].DataF[2]-33) < 0.01) then
    Pass('multi-entrada: a+b [1,2,3]+[10,20,30] -> [11,22,33]')
  else
    Fail('multi-entrada: valores incorrectos');
end;

// Fail-early: shape que no casa con los datos debe lanzar EOnnx ANTES de llamar a ORT
// (evita lectura fuera de rango). Aquí el test PASA si se lanza la excepción.
procedure TestShapeValidation;
var s: TOnnxSession; x: TSingleArray; shp: TInt64Array; raised: Boolean;
begin
  raised := False;
  s := TOnnxSession.Create('model.onnx');
  try
    SetLength(x, 3);   x[0]:=1; x[1]:=2; x[2]:=3;   // 3 datos...
    SetLength(shp, 2); shp[0]:=1; shp[1]:=4;         // ...pero shape [1,4] = 4 elementos
    try
      s.Run(s.InputNames[0], s.OutputNames[0], x, shp);
    except
      on E: EOnnx do raised := True;
    end;
  finally s.Free; end;
  if raised then
    Pass('shape validation: [1,4] con 3 datos -> EOnnx (fail-early)')
  else
    Fail('shape validation: no lanzó EOnnx (posible OOB)');
end;

procedure TestDouble;
var s: TOnnxSession; xd: TDoubleArray; shp: TInt64Array; ins: TOnnxTensorArray;
    names: TStringArray; outs: TOnnxTensorArray;
begin
  s := TOnnxSession.Create('model_f64.onnx');
  try
    SetLength(xd, 3);  xd[0]:=1.5; xd[1]:=2.5; xd[2]:=3.5;
    SetLength(shp, 1); shp[0]:=3;
    SetLength(ins, 1); ins[0] := OnnxDouble('x', xd, shp);
    SetLength(names, 1); names[0] := 'y';
    outs := s.RunMulti(ins, names);
  finally s.Free; end;
  if (Length(outs) = 1) and (outs[0].ElemType = oeDouble) and (Length(outs[0].DataD) = 3)
     and (Abs(outs[0].DataD[0]-2.5) < 1e-9) and (Abs(outs[0].DataD[2]-4.5) < 1e-9) then
    Pass('float64: [1.5,2.5,3.5] + 1.0 -> [2.5,3.5,4.5]')
  else
    Fail('float64: no dio [2.5,3.5,4.5]');
end;

procedure TestInt32Type;
var s: TOnnxSession; xi: TInt32Array; shp: TInt64Array; ins: TOnnxTensorArray;
    names: TStringArray; outs: TOnnxTensorArray;
begin
  s := TOnnxSession.Create('model_i32.onnx');
  try
    SetLength(xi, 3);  xi[0]:=100; xi[1]:=200; xi[2]:=300;
    SetLength(shp, 1); shp[0]:=3;
    SetLength(ins, 1); ins[0] := OnnxInt32('x', xi, shp);
    SetLength(names, 1); names[0] := 'y';
    outs := s.RunMulti(ins, names);
  finally s.Free; end;
  if (Length(outs) = 1) and (outs[0].ElemType = oeInt32) and (Length(outs[0].DataI32) = 3)
     and (outs[0].DataI32[0] = 101) and (outs[0].DataI32[2] = 301) then
    Pass('int32: [100,200,300] + 1 -> [101,201,301]')
  else
    Fail('int32: no dio [101,201,301]');
end;

procedure TestUInt8Type;
var s: TOnnxSession; xu: TByteArray; shp: TInt64Array; ins: TOnnxTensorArray;
    names: TStringArray; outs: TOnnxTensorArray;
begin
  s := TOnnxSession.Create('model_u8.onnx');
  try
    SetLength(xu, 4);  xu[0]:=0; xu[1]:=127; xu[2]:=200; xu[3]:=255;
    SetLength(shp, 1); shp[0]:=4;
    SetLength(ins, 1); ins[0] := OnnxUInt8('x', xu, shp);
    SetLength(names, 1); names[0] := 'y';
    outs := s.RunMulti(ins, names);
  finally s.Free; end;
  if (Length(outs) = 1) and (outs[0].ElemType = oeUInt8) and (Length(outs[0].DataU8) = 4)
     and (outs[0].DataU8[0] = 0) and (outs[0].DataU8[3] = 255) then
    Pass('uint8: identity [0,127,200,255] -> [0,127,200,255]')
  else
    Fail('uint8: no preservó los bytes');
end;

// Conformal regresión: residuos calib [1..10], alpha=0.1 -> q=10 (k=ceil(11*0.9)=10).
procedure TestConformalRegression;
var r: TConformalRegressor; res: TDoubleArray; i: Integer; lo, hi: Double;
begin
  SetLength(res, 10);
  for i := 0 to 9 do res[i] := i + 1;   // [1,2,...,10]
  r := TConformalRegressor.Create(res, 0.1);
  try
    r.Interval(5.0, lo, hi);
    if (Abs(r.Q - 10.0) < 1e-9) and (Abs(lo - (-5.0)) < 1e-9) and (Abs(hi - 15.0) < 1e-9) then
      Pass('conformal reg: q=10, pred 5 -> [-5,15] (cobertura 0.9)')
    else
      Fail(Format('conformal reg: q=%.3f intervalo [%.3f,%.3f]', [r.Q, lo, hi]));
  finally r.Free; end;
end;

// Conformal clasificación (LAC): probTrue calib decreciente, alpha=0.2 -> q=0.4
// (scores 1-p ordenados, k=ceil(10*0.8)=8 -> q=score[8]=0.4). Umbral: p>=0.6.
procedure TestConformalClassification;
var c: TConformalClassifier; calib, probs: TDoubleArray; i: Integer; s: TClassSet;
begin
  SetLength(calib, 9);
  for i := 0 to 8 do calib[i] := 0.95 - i * 0.05;   // 0.95,0.90,...,0.55
  c := TConformalClassifier.Create(calib, 0.2);
  try
    SetLength(probs, 3); probs[0]:=0.65; probs[1]:=0.25; probs[2]:=0.10;
    s := c.PredictionSet(probs);
    if (Abs(c.Q - 0.4) < 1e-9) and (Length(s) = 1) and (s[0] = 0) then
      Pass('conformal clf: q=0.4, probs[.65,.25,.10] -> set {0}')
    else
      Fail(Format('conformal clf: q=%.3f, |set|=%d', [c.Q, Length(s)]));
  finally c.Free; end;
end;

// Borde: alpha muy pequeño con calibración pequeña -> q=+Inf -> set = todas las clases.
procedure TestConformalFullSet;
var c: TConformalClassifier; calib, probs: TDoubleArray; i: Integer; s: TClassSet;
begin
  SetLength(calib, 9);
  for i := 0 to 8 do calib[i] := 0.95 - i * 0.05;
  c := TConformalClassifier.Create(calib, 0.05);   // k=ceil(10*0.95)=10 > 9 -> q=+Inf
  try
    SetLength(probs, 3); probs[0]:=0.65; probs[1]:=0.25; probs[2]:=0.10;
    s := c.PredictionSet(probs);
    if Length(s) = 3 then
      Pass('conformal clf borde: alpha bajo -> set completo {0,1,2}')
    else
      Fail(Format('conformal clf borde: |set|=%d (esperaba 3)', [Length(s)]));
  finally c.Free; end;
end;

// Warm runner: dos inferencias reutilizando el mismo buffer de entrada.
// model.onnx: y = 2a - b + 0.5c.  [1,2,3]->1.5 ; [2,4,6]-> 4-4+3 = 3.0
procedure TestWarmRunner;
var s: TOnnxSession; w: TOnnxWarmRunner; y1, y2: TSingleArray; buf: TSingleArray; shp: TInt64Array;
begin
  s := TOnnxSession.Create('model.onnx');
  try
    SetLength(shp, 2); shp[0]:=1; shp[1]:=3;
    w := TOnnxWarmRunner.Create(s, s.InputNames[0], s.OutputNames[0], shp);
    try
      buf := w.InputBuffer;
      buf[0]:=1; buf[1]:=2; buf[2]:=3;  y1 := w.Infer;
      buf[0]:=2; buf[1]:=4; buf[2]:=6;  y2 := w.Infer;
    finally w.Free; end;
  finally s.Free; end;
  if (Length(y1)=1) and (Abs(y1[0]-1.5) < 0.01)
     and (Length(y2)=1) and (Abs(y2[0]-3.0) < 0.01) then
    Pass('warm runner: reuso de buffer -> 1.5 y 3.0')
  else
    Fail(Format('warm runner: dio %.3f y %.3f (esperaba 1.5 y 3.0)', [y1[0], y2[0]]));
end;

begin
  WriteLn('onnxruntime ', OnnxRuntimeVersion);
  try TestLinearFloat;             except on E: Exception do Fail('float single: ' + E.Message); end;
  try TestInt64;                   except on E: Exception do Fail('int64: ' + E.Message); end;
  try TestMultiOutput;             except on E: Exception do Fail('multi-salida: ' + E.Message); end;
  try TestMultiInput;              except on E: Exception do Fail('multi-entrada: ' + E.Message); end;
  try TestShapeValidation;         except on E: Exception do Fail('shape validation: ' + E.Message); end;
  try TestDouble;                  except on E: Exception do Fail('float64: ' + E.Message); end;
  try TestInt32Type;               except on E: Exception do Fail('int32: ' + E.Message); end;
  try TestUInt8Type;               except on E: Exception do Fail('uint8: ' + E.Message); end;
  try TestConformalRegression;     except on E: Exception do Fail('conformal reg: ' + E.Message); end;
  try TestConformalClassification; except on E: Exception do Fail('conformal clf: ' + E.Message); end;
  try TestConformalFullSet;        except on E: Exception do Fail('conformal borde: ' + E.Message); end;
  try TestWarmRunner;              except on E: Exception do Fail('warm runner: ' + E.Message); end;
  WriteLn('---');
  if failures = 0 then begin WriteLn('TODOS OK (12/12)'); Halt(0); end
  else begin WriteLn(Format('%d fallo(s)', [failures])); Halt(1); end;
end.
