unit uOnnxConformal;
{ ---------------------------------------------------------------------------
  uOnnxConformal — predicción conforme (split conformal) sobre salidas de un
  modelo, con cobertura marginal garantizada 1-alpha bajo intercambiabilidad.

  Es independiente del motor: opera sobre números (residuos / probabilidades),
  así que sirve para cualquier predictor, no solo ONNX. Se apoya en uOnnxRuntime
  únicamente para reutilizar TDoubleArray y EOnnx.

  - TConformalRegressor: intervalos de predicción [ŷ-q, ŷ+q].
  - TConformalClassifier: prediction sets (LAC / score conforme).

  Cuantil conforme (split): con n ejemplos de calibración y nivel alpha,
      k = ceil((n+1)*(1-alpha));  q = scores_ordenados[k]  (1-based)
  Si k > n no hay datos para ese nivel y q = +Inf (intervalo/ set no acotado).

  Licencia: MIT.
  --------------------------------------------------------------------------- }
{$IFDEF FPC}{$mode objfpc}{$H+}{$ENDIF}

interface

uses
  SysUtils, Math, uOnnxRuntime;   // TDoubleArray, EOnnx

type
  TClassSet = array of Integer;   // índices de clase incluidos en el prediction set

  { Cuantil conforme de un conjunto de scores de no-conformidad. }
  function ConformalQuantile(const AScores: TDoubleArray; AAlpha: Double): Double;

type
  { Regresión: intervalos de predicción con cobertura 1-alpha.
    ACalibResiduals = scores de no-conformidad de calibración, típicamente |y - ŷ|. }
  TConformalRegressor = class
  private
    FQ: Double;
  public
    constructor Create(const ACalibResiduals: TDoubleArray; AAlpha: Double);
    procedure Interval(APrediction: Double; out ALo, AHi: Double);
    property Q: Double read FQ;   // semi-anchura del intervalo (+Inf si no acotado)
  end;

  { Clasificación (LAC): prediction sets con cobertura 1-alpha.
    ACalibProbTrue = probabilidad que el modelo asignó a la clase VERDADERA en
    cada ejemplo de calibración (score de no-conformidad = 1 - prob). }
  TConformalClassifier = class
  private
    FQ: Double;
  public
    constructor Create(const ACalibProbTrue: TDoubleArray; AAlpha: Double);
    function PredictionSet(const AProbs: TDoubleArray): TClassSet;
    property Q: Double read FQ;   // umbral de no-conformidad (+Inf => todas las clases)
  end;

implementation

procedure SortAsc(var A: TDoubleArray);   // insertion sort; n de calibración pequeño
var i, j: Integer; key: Double;
begin
  for i := 1 to High(A) do
  begin
    key := A[i]; j := i - 1;
    while (j >= 0) and (A[j] > key) do begin A[j + 1] := A[j]; Dec(j); end;
    A[j + 1] := key;
  end;
end;

function ConformalQuantile(const AScores: TDoubleArray; AAlpha: Double): Double;
var s: TDoubleArray; n, k: Integer;
begin
  n := Length(AScores);
  if n = 0 then raise EOnnx.Create('conformal: sin datos de calibración');
  if (AAlpha <= 0) or (AAlpha >= 1) then
    raise EOnnx.Create('conformal: alpha debe estar en (0,1)');
  s := Copy(AScores, 0, n);
  SortAsc(s);
  k := Ceil((n + 1) * (1 - AAlpha));
  if k > n then Result := Infinity   // nivel no alcanzable con esta calibración
  else Result := s[k - 1];           // k es 1-based
end;

{ TConformalRegressor }
constructor TConformalRegressor.Create(const ACalibResiduals: TDoubleArray; AAlpha: Double);
begin
  inherited Create;
  FQ := ConformalQuantile(ACalibResiduals, AAlpha);
end;

procedure TConformalRegressor.Interval(APrediction: Double; out ALo, AHi: Double);
begin
  if IsInfinite(FQ) then begin ALo := NegInfinity; AHi := Infinity; end
  else begin ALo := APrediction - FQ; AHi := APrediction + FQ; end;
end;

{ TConformalClassifier }
constructor TConformalClassifier.Create(const ACalibProbTrue: TDoubleArray; AAlpha: Double);
var scores: TDoubleArray; i: Integer;
begin
  inherited Create;
  SetLength(scores, Length(ACalibProbTrue));
  for i := 0 to High(ACalibProbTrue) do
    scores[i] := 1.0 - ACalibProbTrue[i];   // no-conformidad = 1 - prob(clase verdadera)
  FQ := ConformalQuantile(scores, AAlpha);
end;

function TConformalClassifier.PredictionSet(const AProbs: TDoubleArray): TClassSet;
var k, m: Integer;
begin
  SetLength(Result, Length(AProbs));
  m := 0;
  for k := 0 to High(AProbs) do
    if (1.0 - AProbs[k]) <= FQ then   // clase k dentro del set si su no-conformidad <= q
    begin
      Result[m] := k; Inc(m);
    end;
  SetLength(Result, m);
end;

end.
