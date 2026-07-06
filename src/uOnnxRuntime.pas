unit uOnnxRuntime;
{ ---------------------------------------------------------------------------
  uOnnxRuntime — inferencia ONNX desde Object Pascal (Free Pascal / Delphi)
  llamando a la C API de onnxruntime (onnxruntime.dll / libonnxruntime.so)
  de forma NATIVA, sin compilador C (sin CGO).

  v0.2: soporte multi-entrada/multi-salida y tensores float32 e int64.

  La OrtApi es una tabla de punteros a función; se accede por índice tomado del
  header oficial onnxruntime_c_api.h (verificado contra onnxruntime 1.27, API v28).

  Verificado con FPC 3.2 en Windows (Win64) y Linux (Debian 13, x86_64), tests 3/3.
  Delphi: mismo enfoque (tipos estándar + stdcall).
  Licencia: MIT.
  --------------------------------------------------------------------------- }
{$IFDEF FPC}{$mode objfpc}{$H+}{$ENDIF}

interface

uses
  SysUtils;

type
  EOnnx = class(Exception);
  TSingleArray = array of Single;
  TInt64Array  = array of Int64;
  TDoubleArray = array of Double;
  TInt32Array  = array of Int32;
  TByteArray   = array of Byte;
  TStringArray = array of string;

  { Tipos de elemento soportados. Superficie mínima orientada a ML clásico/edge:
    float32/float64 (regresión), int32/int64 (etiquetas, índices), uint8 (imágenes).
    Añadir uno nuevo = una entrada en cada tabla (ONNX_CODE/ELEM_SIZE) + su array. }
  TOnnxElemType = (oeFloat, oeInt64, oeDouble, oeInt32, oeUInt8);

  { Tensor genérico: los datos viven en el array correspondiente a ElemType. }
  TOnnxTensor = record
    Name: string;
    ElemType: TOnnxElemType;
    Shape: TInt64Array;
    DataF:   TSingleArray;   // oeFloat  (float32)
    DataI:   TInt64Array;    // oeInt64
    DataD:   TDoubleArray;   // oeDouble (float64)
    DataI32: TInt32Array;    // oeInt32
    DataU8:  TByteArray;     // oeUInt8
  end;
  TOnnxTensorArray = array of TOnnxTensor;

  { TOnnxSession — una sesión de inferencia sobre un modelo .onnx cargado. }
  TOnnxSession = class
  private
    FEnv, FOpts, FSession, FMemInfo: Pointer;
    function GetIOName(GetCountIdx, GetNameIdx: Integer): TStringArray;
  public
    constructor Create(const AModelPath: string);
    destructor Destroy; override;

    function InputNames: TStringArray;
    function OutputNames: TStringArray;

    { Inferencia general: N tensores de entrada -> M tensores de salida (por nombre).
      Cada salida trae su ElemType, Shape y los datos en DataF o DataI. }
    function RunMulti(const AInputs: TOnnxTensorArray; const AOutputNames: TStringArray): TOnnxTensorArray;

    { Conveniencia: un tensor float32 de entrada -> un tensor float32 de salida. }
    function Run(const AInputName, AOutputName: string;
                 const AData: TSingleArray; const AShape: TInt64Array): TSingleArray;
  end;

  { TOnnxWarmRunner — ruta "caliente" para inferencia repetida con shape fijo.
    Crea el OrtValue de entrada UNA vez sobre un buffer reutilizable y precomputa
    los nombres; cada Infer reutiliza ambos -> sin asignaciones de entrada por
    llamada, menor latencia y menos jitter. Caso: 1 entrada float32 -> 1 salida float32.
    Uso: escribe en InputBuffer[i] (no cambies su tamaño) y llama a Infer. }
  TOnnxWarmRunner = class
  private
    FSession: TOnnxSession;
    FInVal: Pointer;
    FBuf: TSingleArray;
    FShape: TInt64Array;
    FInNameA, FOutNameA: AnsiString;
    FInNameP, FOutNameP: PAnsiChar;
  public
    constructor Create(ASession: TOnnxSession; const AInputName, AOutputName: string;
                       const AShape: TInt64Array);
    destructor Destroy; override;
    function Infer: TSingleArray;
    property InputBuffer: TSingleArray read FBuf;
  end;

{ Constructores de tensor de entrada. }
function OnnxFloat(const AName: string; const AData: TSingleArray; const AShape: TInt64Array): TOnnxTensor;
function OnnxInt64(const AName: string; const AData: TInt64Array; const AShape: TInt64Array): TOnnxTensor;
function OnnxDouble(const AName: string; const AData: TDoubleArray; const AShape: TInt64Array): TOnnxTensor;
function OnnxInt32(const AName: string; const AData: TInt32Array; const AShape: TInt64Array): TOnnxTensor;
function OnnxUInt8(const AName: string; const AData: TByteArray; const AShape: TInt64Array): TOnnxTensor;

{ Versión de la librería nativa onnxruntime cargada. }
function OnnxRuntimeVersion: string;

implementation

{$IFNDEF FPC}
type PtrUInt = NativeUInt;   // FPC lo trae de serie; Delphi usa NativeUInt
{$ENDIF}

// ---------------------------------------------------------------------------
// C API — tipos y binding por índice
// ---------------------------------------------------------------------------
type
  PPAnsiChar = ^PAnsiChar;

  TGetApi = function(version: LongWord): Pointer; stdcall;
  TGetVer = function: PAnsiChar; stdcall;
  TOrtApiBase = record
    GetApi: TGetApi;
    GetVersionString: TGetVer;
  end;
  POrtApiBase = ^TOrtApiBase;
  TApiArr = array[0..511] of Pointer;
  PApiArr = ^TApiArr;

  TFnCreateEnv      = function(lvl: LongInt; logid: PAnsiChar; out env: Pointer): Pointer; stdcall;
  TFnCreateSessOpts = function(out opts: Pointer): Pointer; stdcall;
  // ORTCHAR_T = wchar_t en Windows (UTF-16), char en el resto (UTF-8)
  {$IFDEF MSWINDOWS}
  TFnCreateSession  = function(env: Pointer; path: PWideChar; opts: Pointer; out sess: Pointer): Pointer; stdcall;
  {$ELSE}
  TFnCreateSession  = function(env: Pointer; path: PAnsiChar; opts: Pointer; out sess: Pointer): Pointer; stdcall;
  {$ENDIF}
  TFnCreateCpuMem   = function(alloc: LongInt; mem: LongInt; out mi: Pointer): Pointer; stdcall;
  TFnCreateTensor   = function(mi: Pointer; data: Pointer; dataLen: PtrUInt; shape: PInt64;
                               shapeLen: PtrUInt; elemType: LongInt; out val: Pointer): Pointer; stdcall;
  TFnRun            = function(sess: Pointer; ro: Pointer; inNames: PPAnsiChar; inputs: PPointer;
                               inLen: PtrUInt; outNames: PPAnsiChar; outLen: PtrUInt; outputs: PPointer): Pointer; stdcall;
  TFnGetTensorData  = function(val: Pointer; out data: Pointer): Pointer; stdcall;
  TFnGetErrMsg      = function(status: Pointer): PAnsiChar; stdcall;
  TFnRelease        = procedure(h: Pointer); stdcall;
  TFnSessGetCount   = function(sess: Pointer; out count: PtrUInt): Pointer; stdcall;
  TFnSessGetName    = function(sess: Pointer; index: PtrUInt; alloc: Pointer; out name: PAnsiChar): Pointer; stdcall;
  TFnGetAllocator   = function(out alloc: Pointer): Pointer; stdcall;
  TFnAllocatorFree  = function(alloc: Pointer; p: Pointer): Pointer; stdcall;
  TFnGetTensorShape = function(val: Pointer; out info: Pointer): Pointer; stdcall;
  TFnGetElemCount   = function(info: Pointer; out count: PtrUInt): Pointer; stdcall;
  TFnGetElemType    = function(info: Pointer; out etype: LongInt): Pointer; stdcall;
  TFnGetDimsCount   = function(info: Pointer; out n: PtrUInt): Pointer; stdcall;
  TFnGetDims        = function(info: Pointer; dims: PInt64; dimsLen: PtrUInt): Pointer; stdcall;

const
  // Índices 1-based del struct OrtApi (onnxruntime_c_api.h). api^[idx-1].
  I_GetErrorMessage   = 3;
  I_CreateEnv         = 4;
  I_CreateSession     = 8;
  I_Run               = 10;
  I_CreateSessOpts    = 11;
  I_SessGetInputCount = 31;
  I_SessGetOutputCount= 32;
  I_SessGetInputName  = 37;
  I_SessGetOutputName = 38;
  I_CreateTensor      = 50;
  I_GetTensorData     = 52;
  I_GetTensorElemType = 61;
  I_GetDimsCount      = 62;
  I_GetDims           = 63;
  I_GetShapeElemCount = 65;
  I_GetTensorTypeShp  = 66;
  I_CreateCpuMem      = 70;
  I_AllocatorFree     = 77;
  I_GetAllocatorDef   = 79;
  I_ReleaseEnv        = 93;
  I_ReleaseMemInfo    = 95;
  I_ReleaseSession    = 96;
  I_ReleaseValue      = 97;
  I_ReleaseTensorInfo = 100;
  I_ReleaseSessOpts   = 101;

  // Códigos ONNXTensorElementDataType (onnxruntime_c_api.h)
  ONNX_FLOAT      = 1;
  ONNX_UINT8      = 2;
  ONNX_INT32      = 6;
  ONNX_INT64      = 7;
  ONNX_DOUBLE     = 11;
  ORT_ARENA_ALLOC = 1;
  ORT_MEM_DEFAULT = 0;
  ORT_LOG_WARNING = 2;
  ORT_API_VERSION = 16;

function OrtGetApiBase: POrtApiBase; stdcall; external {$IFDEF MSWINDOWS}'onnxruntime.dll'{$ELSE}'libonnxruntime.so'{$ENDIF} name 'OrtGetApiBase';

var
  gApi: PApiArr = nil;

function Fn(idx: Integer): Pointer; inline;
begin
  Fn := gApi^[idx - 1];
end;

procedure InitApi;
var base: POrtApiBase; p: Pointer;
begin
  if gApi <> nil then Exit;
  base := OrtGetApiBase();
  p := base^.GetApi(ORT_API_VERSION);
  if p = nil then
    raise EOnnx.Create('OrtGetApiBase().GetApi devolvió nil (versión de API no soportada por la DLL)');
  gApi := PApiArr(p);
end;

procedure Check(st: Pointer; const ctx: string);
begin
  if st <> nil then
    raise EOnnx.CreateFmt('%s: %s', [ctx, string(TFnGetErrMsg(Fn(I_GetErrorMessage))(st))]);
end;

function OnnxRuntimeVersion: string;
begin
  Result := string(OrtGetApiBase()^.GetVersionString());
end;

function OnnxFloat(const AName: string; const AData: TSingleArray; const AShape: TInt64Array): TOnnxTensor;
begin
  Result.Name := AName; Result.ElemType := oeFloat;
  Result.Shape := AShape; Result.DataF := AData; Result.DataI := nil;
end;

function OnnxInt64(const AName: string; const AData: TInt64Array; const AShape: TInt64Array): TOnnxTensor;
begin
  Result.Name := AName; Result.ElemType := oeInt64;
  Result.Shape := AShape; Result.DataI := AData;
end;

function OnnxDouble(const AName: string; const AData: TDoubleArray; const AShape: TInt64Array): TOnnxTensor;
begin
  Result.Name := AName; Result.ElemType := oeDouble;
  Result.Shape := AShape; Result.DataD := AData;
end;

function OnnxInt32(const AName: string; const AData: TInt32Array; const AShape: TInt64Array): TOnnxTensor;
begin
  Result.Name := AName; Result.ElemType := oeInt32;
  Result.Shape := AShape; Result.DataI32 := AData;
end;

function OnnxUInt8(const AName: string; const AData: TByteArray; const AShape: TInt64Array): TOnnxTensor;
begin
  Result.Name := AName; Result.ElemType := oeUInt8;
  Result.Shape := AShape; Result.DataU8 := AData;
end;

// --- Tablas por tipo (código ONNX y tamaño de elemento en bytes) ---
function ElemOnnxCode(et: TOnnxElemType): LongInt;
begin
  case et of
    oeFloat:  Result := ONNX_FLOAT;
    oeInt64:  Result := ONNX_INT64;
    oeDouble: Result := ONNX_DOUBLE;
    oeInt32:  Result := ONNX_INT32;
    oeUInt8:  Result := ONNX_UINT8;
  else raise EOnnx.Create('tipo de elemento no soportado');
  end;
end;

function ElemSize(et: TOnnxElemType): Integer;
begin
  case et of
    oeFloat:  Result := SizeOf(Single);
    oeInt64:  Result := SizeOf(Int64);
    oeDouble: Result := SizeOf(Double);
    oeInt32:  Result := SizeOf(Int32);
    oeUInt8:  Result := SizeOf(Byte);
  else raise EOnnx.Create('tipo de elemento no soportado');
  end;
end;

// Puntero al buffer de datos y nº de elementos del array activo según ElemType.
procedure InputData(const T: TOnnxTensor; out p: Pointer; out nElem: Integer);
begin
  case T.ElemType of
    oeFloat:  begin nElem := Length(T.DataF);   if nElem > 0 then p := @T.DataF[0]   else p := nil; end;
    oeInt64:  begin nElem := Length(T.DataI);   if nElem > 0 then p := @T.DataI[0]   else p := nil; end;
    oeDouble: begin nElem := Length(T.DataD);   if nElem > 0 then p := @T.DataD[0]   else p := nil; end;
    oeInt32:  begin nElem := Length(T.DataI32); if nElem > 0 then p := @T.DataI32[0] else p := nil; end;
    oeUInt8:  begin nElem := Length(T.DataU8);  if nElem > 0 then p := @T.DataU8[0]  else p := nil; end;
  else raise EOnnx.Create('tipo de elemento no soportado');
  end;
end;

// Mapea el código ONNX de una salida a nuestro TOnnxElemType (o falla claro).
function OnnxCodeToElem(onnxCode: LongInt): TOnnxElemType;
begin
  case onnxCode of
    ONNX_FLOAT:  Result := oeFloat;
    ONNX_INT64:  Result := oeInt64;
    ONNX_DOUBLE: Result := oeDouble;
    ONNX_INT32:  Result := oeInt32;
    ONNX_UINT8:  Result := oeUInt8;
  else raise EOnnx.CreateFmt('tipo de salida ONNX no soportado (code=%d)', [onnxCode]);
  end;
end;

// Copia count elementos desde dataPtr al array del tensor según su ElemType.
procedure ReadOutput(var T: TOnnxTensor; count: PtrUInt; dataPtr: Pointer);
begin
  case T.ElemType of
    oeFloat:  begin SetLength(T.DataF, count);   if count > 0 then Move(dataPtr^, T.DataF[0],   count * SizeOf(Single)); end;
    oeInt64:  begin SetLength(T.DataI, count);   if count > 0 then Move(dataPtr^, T.DataI[0],   count * SizeOf(Int64));  end;
    oeDouble: begin SetLength(T.DataD, count);   if count > 0 then Move(dataPtr^, T.DataD[0],   count * SizeOf(Double)); end;
    oeInt32:  begin SetLength(T.DataI32, count); if count > 0 then Move(dataPtr^, T.DataI32[0], count * SizeOf(Int32));  end;
    oeUInt8:  begin SetLength(T.DataU8, count);  if count > 0 then Move(dataPtr^, T.DataU8[0],  count * SizeOf(Byte));   end;
  else raise EOnnx.Create('tipo de elemento no soportado');
  end;
end;

// Nº de elementos que implica una forma (shape). Shape vacío = escalar = 1 elemento.
// Rechaza dimensiones negativas (una dimensión dinámica -1 no es válida para una entrada concreta).
function ShapeElemCount(const AShape: TInt64Array): Int64;
var i: Integer;
begin
  Result := 1;
  for i := 0 to High(AShape) do
  begin
    if AShape[i] < 0 then
      raise EOnnx.CreateFmt('shape con dimensión negativa (%d) en posición %d', [AShape[i], i]);
    Result := Result * AShape[i];
  end;
end;

function ShapeToStr(const AShape: TInt64Array): string;
var i: Integer;
begin
  Result := '[';
  for i := 0 to High(AShape) do
  begin
    if i > 0 then Result := Result + ',';
    Result := Result + IntToStr(AShape[i]);
  end;
  Result := Result + ']';
end;

// ---------------------------------------------------------------------------
// TOnnxSession
// ---------------------------------------------------------------------------
constructor TOnnxSession.Create(const AModelPath: string);
{$IFDEF MSWINDOWS}var wpath: UnicodeString;{$ELSE}var apath: AnsiString;{$ENDIF}
begin
  inherited Create;
  InitApi;
  Check(TFnCreateEnv(Fn(I_CreateEnv))(ORT_LOG_WARNING, 'uOnnxRuntime', FEnv), 'CreateEnv');
  Check(TFnCreateSessOpts(Fn(I_CreateSessOpts))(FOpts), 'CreateSessionOptions');
  {$IFDEF MSWINDOWS}
  wpath := UnicodeString(AModelPath);
  Check(TFnCreateSession(Fn(I_CreateSession))(FEnv, PWideChar(wpath), FOpts, FSession), 'CreateSession');
  {$ELSE}
  apath := AnsiString(AModelPath);   // UTF-8
  Check(TFnCreateSession(Fn(I_CreateSession))(FEnv, PAnsiChar(apath), FOpts, FSession), 'CreateSession');
  {$ENDIF}
  Check(TFnCreateCpuMem(Fn(I_CreateCpuMem))(ORT_ARENA_ALLOC, ORT_MEM_DEFAULT, FMemInfo), 'CreateCpuMemoryInfo');
end;

destructor TOnnxSession.Destroy;
begin
  if FMemInfo <> nil then TFnRelease(Fn(I_ReleaseMemInfo))(FMemInfo);
  if FSession <> nil then TFnRelease(Fn(I_ReleaseSession))(FSession);
  if FOpts    <> nil then TFnRelease(Fn(I_ReleaseSessOpts))(FOpts);
  if FEnv     <> nil then TFnRelease(Fn(I_ReleaseEnv))(FEnv);
  inherited Destroy;
end;

function TOnnxSession.GetIOName(GetCountIdx, GetNameIdx: Integer): TStringArray;
var
  count, i: PtrUInt;
  alloc, namePtr: Pointer;
  cname: PAnsiChar;
begin
  Check(TFnGetAllocator(Fn(I_GetAllocatorDef))(alloc), 'GetAllocatorWithDefaultOptions');
  Check(TFnSessGetCount(Fn(GetCountIdx))(FSession, count), 'SessionGet*Count');
  SetLength(Result, count);
  for i := 0 to count - 1 do
  begin
    Check(TFnSessGetName(Fn(GetNameIdx))(FSession, i, alloc, cname), 'SessionGet*Name');
    Result[i] := string(cname);
    namePtr := cname;
    TFnAllocatorFree(Fn(I_AllocatorFree))(alloc, namePtr);
  end;
end;

function TOnnxSession.InputNames: TStringArray;
begin Result := GetIOName(I_SessGetInputCount, I_SessGetInputName); end;

function TOnnxSession.OutputNames: TStringArray;
begin Result := GetIOName(I_SessGetOutputCount, I_SessGetOutputName); end;

function TOnnxSession.RunMulti(const AInputs: TOnnxTensorArray;
  const AOutputNames: TStringArray): TOnnxTensorArray;
var
  nIn, nOut, i, j: Integer;
  inVals, outVals: array of Pointer;
  inNameP, outNameP: array of PAnsiChar;
  inNameA, outNameA: array of AnsiString;
  dataPtr, shapeInfo: Pointer;
  count, dimCount: PtrUInt;
  etype: LongInt;
  nElem: Integer;
begin
  nIn := Length(AInputs);
  nOut := Length(AOutputNames);
  SetLength(inVals, nIn);  SetLength(inNameP, nIn);  SetLength(inNameA, nIn);
  SetLength(outVals, nOut);SetLength(outNameP, nOut);SetLength(outNameA, nOut);
  SetLength(Result, nOut);
  for i := 0 to nIn - 1 do inVals[i] := nil;
  for j := 0 to nOut - 1 do outVals[j] := nil;

  try
    // crear tensores de entrada (camino único por tabla de tipos)
    for i := 0 to nIn - 1 do
    begin
      InputData(AInputs[i], dataPtr, nElem);
      // Verificación de shape (fail-early): el nº de datos debe casar con la forma.
      // Evita que ORT lea fuera del buffer si shape y datos no concuerdan.
      if ShapeElemCount(AInputs[i].Shape) <> nElem then
        raise EOnnx.CreateFmt('entrada "%s": shape %s implica %d elementos, pero se pasaron %d',
          [AInputs[i].Name, ShapeToStr(AInputs[i].Shape), ShapeElemCount(AInputs[i].Shape), nElem]);
      Check(TFnCreateTensor(Fn(I_CreateTensor))(FMemInfo, dataPtr,
            PtrUInt(nElem) * PtrUInt(ElemSize(AInputs[i].ElemType)), @AInputs[i].Shape[0],
            Length(AInputs[i].Shape), ElemOnnxCode(AInputs[i].ElemType), inVals[i]), 'CreateTensor');
      inNameA[i] := AnsiString(AInputs[i].Name);
      inNameP[i] := PAnsiChar(inNameA[i]);
    end;
    for j := 0 to nOut - 1 do
    begin
      outNameA[j] := AnsiString(AOutputNames[j]);
      outNameP[j] := PAnsiChar(outNameA[j]);
    end;

    Check(TFnRun(Fn(I_Run))(FSession, nil, @inNameP[0], @inVals[0], nIn,
          @outNameP[0], nOut, @outVals[0]), 'Run');

    // leer cada salida (tipo, forma y datos)
    for j := 0 to nOut - 1 do
    begin
      Result[j].Name := AOutputNames[j];
      Check(TFnGetTensorShape(Fn(I_GetTensorTypeShp))(outVals[j], shapeInfo), 'GetTensorTypeAndShape');
      try
        Check(TFnGetElemCount(Fn(I_GetShapeElemCount))(shapeInfo, count), 'GetTensorShapeElementCount');
        Check(TFnGetElemType(Fn(I_GetTensorElemType))(shapeInfo, etype), 'GetTensorElementType');
        Check(TFnGetDimsCount(Fn(I_GetDimsCount))(shapeInfo, dimCount), 'GetDimensionsCount');
        SetLength(Result[j].Shape, dimCount);
        if dimCount > 0 then
          Check(TFnGetDims(Fn(I_GetDims))(shapeInfo, @Result[j].Shape[0], dimCount), 'GetDimensions');
      finally
        TFnRelease(Fn(I_ReleaseTensorInfo))(shapeInfo);
      end;
      Check(TFnGetTensorData(Fn(I_GetTensorData))(outVals[j], dataPtr), 'GetTensorMutableData');
      Result[j].ElemType := OnnxCodeToElem(etype);
      ReadOutput(Result[j], count, dataPtr);
    end;
  finally
    for j := 0 to nOut - 1 do if outVals[j] <> nil then TFnRelease(Fn(I_ReleaseValue))(outVals[j]);
    for i := 0 to nIn - 1 do  if inVals[i]  <> nil then TFnRelease(Fn(I_ReleaseValue))(inVals[i]);
  end;
end;

function TOnnxSession.Run(const AInputName, AOutputName: string;
  const AData: TSingleArray; const AShape: TInt64Array): TSingleArray;
var
  ins: TOnnxTensorArray;
  names: TStringArray;
  outs: TOnnxTensorArray;
begin
  SetLength(ins, 1);  ins[0] := OnnxFloat(AInputName, AData, AShape);
  SetLength(names, 1); names[0] := AOutputName;
  outs := RunMulti(ins, names);
  Result := outs[0].DataF;
end;

// ---------------------------------------------------------------------------
// TOnnxWarmRunner
// ---------------------------------------------------------------------------
constructor TOnnxWarmRunner.Create(ASession: TOnnxSession;
  const AInputName, AOutputName: string; const AShape: TInt64Array);
var nElem: Int64;
begin
  inherited Create;
  FSession := ASession;
  FShape := Copy(AShape, 0, Length(AShape));
  nElem := ShapeElemCount(FShape);
  SetLength(FBuf, nElem);   // no se vuelve a redimensionar: el puntero @FBuf[0] queda estable
  FInNameA  := AnsiString(AInputName);  FInNameP  := PAnsiChar(FInNameA);
  FOutNameA := AnsiString(AOutputName); FOutNameP := PAnsiChar(FOutNameA);
  // El OrtValue referencia (no copia) @FBuf[0]; mutar FBuf entre llamadas basta.
  Check(TFnCreateTensor(Fn(I_CreateTensor))(FSession.FMemInfo, @FBuf[0],
        PtrUInt(nElem) * SizeOf(Single), @FShape[0], Length(FShape),
        ONNX_FLOAT, FInVal), 'CreateTensor(warm)');
end;

destructor TOnnxWarmRunner.Destroy;
begin
  if FInVal <> nil then TFnRelease(Fn(I_ReleaseValue))(FInVal);
  inherited Destroy;
end;

function TOnnxWarmRunner.Infer: TSingleArray;
var
  inNameP, outNameP: array[0..0] of PAnsiChar;
  inVals, outVals: array[0..0] of Pointer;
  shapeInfo, dataPtr: Pointer;
  count: PtrUInt;
begin
  inNameP[0] := FInNameP; outNameP[0] := FOutNameP;
  inVals[0] := FInVal;    outVals[0] := nil;
  Check(TFnRun(Fn(I_Run))(FSession.FSession, nil, @inNameP[0], @inVals[0], 1,
        @outNameP[0], 1, @outVals[0]), 'Run(warm)');
  try
    Check(TFnGetTensorShape(Fn(I_GetTensorTypeShp))(outVals[0], shapeInfo), 'GetTensorTypeAndShape');
    try
      Check(TFnGetElemCount(Fn(I_GetShapeElemCount))(shapeInfo, count), 'GetTensorShapeElementCount');
    finally
      TFnRelease(Fn(I_ReleaseTensorInfo))(shapeInfo);
    end;
    Check(TFnGetTensorData(Fn(I_GetTensorData))(outVals[0], dataPtr), 'GetTensorMutableData');
    SetLength(Result, count);
    if count > 0 then Move(dataPtr^, Result[0], count * SizeOf(Single));
  finally
    if outVals[0] <> nil then TFnRelease(Fn(I_ReleaseValue))(outVals[0]);
  end;
end;

end.
