program bench;
{ Benchmark de latencia de inferencia de la librería uOnnxRuntime.
  Mide el coste POR LLAMADA del ciclo completo (crear tensor de entrada + Run +
  copiar salida), que es lo que experimenta el usuario. Reporta min/mean/p50/p95/p99/max
  en microsegundos y throughput.

  Uso:  bench [model.onnx] [N] [p99_max_us]
        (por defecto model.onnx, N=20000; si se da p99_max_us y el p99 WARM lo
         supera, el proceso sale con código 1 -> puerta de regresión para CI).

  Compara la ruta "cold" (Run: crea tensor + nombres por llamada) con la "warm"
  (TOnnxWarmRunner: reutiliza el OrtValue de entrada y los nombres). La warm debe
  dar menor latencia y menos jitter.

  Nota: la entrada por defecto es float [1,3] (modelo de prueba). Para otro modelo,
  ajusta x/shp a la forma de su entrada.

  Temporización: QueryPerformanceCounter en Windows, clock_gettime(CLOCK_MONOTONIC) en
  Linux (ambos sub-µs). Se resta el contador en Int64 ANTES de convertir a µs (evita
  pérdida de precisión con contadores grandes). }
{$IFDEF FPC}{$mode objfpc}{$H+}{$ENDIF}
uses
  {$IFDEF MSWINDOWS}Windows,{$ENDIF}
  {$IFDEF UNIX}unixtype, linux,{$ENDIF}
  SysUtils, uOnnxRuntime;

var
  gFreq: Int64;

{$IFDEF MSWINDOWS}
function Ticks: Int64; begin QueryPerformanceCounter(Result); end;
{$ELSE}
function Ticks: Int64;
var ts: TTimeSpec;
begin
  clock_gettime(CLOCK_MONOTONIC, @ts);   // reloj monotónico en nanosegundos
  Ticks := Int64(ts.tv_sec) * 1000000000 + Int64(ts.tv_nsec);
end;
{$ENDIF}

procedure QSort(var a: array of Double; lo, hi: Integer);
var i, j: Integer; p, t: Double;
begin
  i := lo; j := hi; p := a[(lo + hi) div 2];
  repeat
    while a[i] < p do Inc(i);
    while a[j] > p do Dec(j);
    if i <= j then
    begin t := a[i]; a[i] := a[j]; a[j] := t; Inc(i); Dec(j); end;
  until i > j;
  if lo < j then QSort(a, lo, j);
  if i < hi then QSort(a, i, hi);
end;

function Pct(const a: array of Double; p: Double): Double;
var idx: Integer;
begin
  idx := Trunc(p * (Length(a) - 1));
  if idx < 0 then idx := 0;
  if idx > High(a) then idx := High(a);
  Pct := a[idx];
end;

// Calcula estadísticos sobre us[] (los ordena) y devuelve el p99. mean por referencia.
function Report(const tag: string; var us: array of Double; out p99: Double): Double;
var i: Integer; sum: Double;
begin
  QSort(us, 0, High(us));
  sum := 0;
  for i := 0 to High(us) do sum := sum + us[i];
  p99 := Pct(us, 0.99);
  WriteLn(Format('  [%-4s] min=%.2f mean=%.2f p50=%.2f p95=%.2f p99=%.2f max=%.2f  jitter(max-min)=%.2f',
    [tag, us[0], sum / Length(us), Pct(us, 0.50), Pct(us, 0.95), p99, us[High(us)], us[High(us)] - us[0]]));
  Report := sum / Length(us);
end;

var
  sess: TOnnxSession;
  warm: TOnnxWarmRunner;
  modelPath, inName, outName: string;
  x: TSingleArray; shp: TInt64Array; buf: TSingleArray;
  N, warmup, i: Integer;
  usC, usW: array of Double;
  c0, c1: Int64;
  p99C, p99W, thr: Double;
  haveThr, warmOK: Boolean;
  fs: TFormatSettings;
begin
  {$IFDEF MSWINDOWS}QueryPerformanceFrequency(gFreq);{$ELSE}gFreq := 1000000000;{$ENDIF}
  modelPath := 'model.onnx';
  N := 20000; warmup := 500;
  thr := 0; haveThr := False;
  if ParamCount >= 1 then modelPath := ParamStr(1);
  if ParamCount >= 2 then N := StrToIntDef(ParamStr(2), N);
  fs := DefaultFormatSettings; fs.DecimalSeparator := '.';   // umbral con punto, sin depender del locale
  if ParamCount >= 3 then begin thr := StrToFloatDef(ParamStr(3), 0, fs); haveThr := thr > 0; end;

  SetLength(usC, N); SetLength(usW, N);
  SetLength(x, 3);   x[0] := 1; x[1] := 2; x[2] := 3;
  SetLength(shp, 2); shp[0] := 1; shp[1] := 3;

  sess := TOnnxSession.Create(modelPath);
  try
    inName := sess.InputNames[0];
    outName := sess.OutputNames[0];
    WriteLn(Format('onnxruntime %s | modelo %s | in=%s out=%s | N=%d (warmup %d)',
      [OnnxRuntimeVersion, modelPath, inName, outName, N, warmup]));

    // --- COLD: Run crea tensor + nombres por llamada ---
    for i := 1 to warmup do sess.Run(inName, outName, x, shp);
    for i := 0 to N - 1 do
    begin
      c0 := Ticks; sess.Run(inName, outName, x, shp); c1 := Ticks;
      usC[i] := (c1 - c0) * 1000000.0 / gFreq;
    end;

    // --- WARM: reutiliza OrtValue de entrada + nombres ---
    warmOK := True;
    try
      warm := TOnnxWarmRunner.Create(sess, inName, outName, shp);
      try
        buf := warm.InputBuffer;
        if Length(buf) >= 3 then begin buf[0]:=1; buf[1]:=2; buf[2]:=3; end;
        for i := 1 to warmup do warm.Infer;
        for i := 0 to N - 1 do
        begin
          c0 := Ticks; warm.Infer; c1 := Ticks;
          usW[i] := (c1 - c0) * 1000000.0 / gFreq;
        end;
      finally warm.Free; end;
    except
      on E: Exception do begin warmOK := False; WriteLn('  [warn] warm no aplica a este modelo: ', E.Message); end;
    end;
  finally
    sess.Free;
  end;

  WriteLn('--- latencia por inferencia (microsegundos) ---');
  Report('cold', usC, p99C);
  if warmOK then Report('warm', usW, p99W) else p99W := p99C;

  if haveThr then
  begin
    WriteLn(Format('--- contrato: p99 warm (%.2f) <= umbral (%.2f) ? ---', [p99W, thr]));
    if p99W > thr then
    begin
      WriteLn('  FALLO: p99 warm supera el umbral'); Halt(1);
    end
    else WriteLn('  OK');
  end;
end.
