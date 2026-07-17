# Evaluacion local de audio

El corpus y sus reportes contienen datos privados y viven fuera de Git. La
ruta operativa recomendada es `D:\AudioBench\aurelius-2026`; los modelos
verificados viven en `D:\Models`.

## Entorno Windows CPU

```powershell
.\.venv\Scripts\python.exe -m pip install -c constraints/windows-cpu.txt -e ".[dev,evaluation]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m speechtotext.evaluation corpus init-report-key --dataset-root D:\AudioBench\aurelius-2026 --repo-root C:\src\speechtotext --output D:\AudioBench\aurelius-2026\secrets\report-ref.key
```

El replay genera su snapshot de entorno despues de auditar el corpus y leer la
clave por `DatasetSecurityEvidence.read_report_ref_key`; no abras la clave con
`Path.read_bytes()` en scripts auxiliares.

## Dataset privado

`manifest.json` usa `speechtotext.corpus/v1`. Cada entrada contiene una ruta
relativa y SHA-256 por cada asset primario/derivado/backup bajo `dataset_root`,
sesion, fecha, condicion, transcript,
duracion declarada, regiones de voz, labels, procedencia, consentimiento/licencia y
`retention_until`. La retencion inicial es 180 dias. El loader rechaza rutas
fuera del dataset, manifests dentro del repo y audio cuyo SHA-256 cambio.

Antes de listar, evaluar, renovar o purgar, el audit local debe demostrar que
el dataset esta fuera de Git, su DACL es current-user-only y el cifrado en
reposo configurado esta activo. Una opcion declarada sin evidencia del OS falla
cerrado. Los reportes de replay y mensajes de consola no contienen audio,
transcripts, ids directos, SHA-256 crudos ni paths absolutos. Las referencias de
drill-down son HMAC y requieren una clave privada de al menos 32 bytes guardada
fuera de Git; con la misma clave siguen siendo deterministas y una rotacion las
vuelve no enlazables.

Cada particion live que se presenta a un gate suma al menos 30 minutos (target
operativo 30-45), repartidos en al menos tres fechas y tres sesiones; por eso el
manifest completo reserva al menos 90 minutos y nueve fechas antes del split.
Cada particion incluye `clean`, `noise` y `silence`, ademas de voz
limpia/rapida/baja, wake continuo y con pausa,
terminos tecnicos espanol/ingles, teclado/ventilador/ruido domestico, silencio,
sonidos no vocales, otras voces, TV, replay y TTS consentido. Los splits se
hacen por `recorded_on`; nunca por clips aleatorios.

Estructura:

```text
D:\AudioBench\aurelius-2026\
  manifest.json
  clips\
    day1-001.wav
  reports\
  secrets\
    report-ref.key
```

## Retencion

`purge-expired` solo muestra el plan salvo que se agregue `--confirm`. Renovar
tambien exige confirmacion explicita y solo permite extender fechas. Las tres
operaciones trabajan exclusivamente sobre assets declarados:

```powershell
.\.venv\Scripts\python.exe -m speechtotext.evaluation corpus list --manifest D:\AudioBench\aurelius-2026\manifest.json --dataset-root D:\AudioBench\aurelius-2026 --repo-root C:\src\speechtotext --report-ref-key-file D:\AudioBench\aurelius-2026\secrets\report-ref.key
.\.venv\Scripts\python.exe -m speechtotext.evaluation corpus renew --manifest D:\AudioBench\aurelius-2026\manifest.json --dataset-root D:\AudioBench\aurelius-2026 --repo-root C:\src\speechtotext --report-ref-key-file D:\AudioBench\aurelius-2026\secrets\report-ref.key --clip-id day1-001 --until 2027-07-12 --confirm
.\.venv\Scripts\python.exe -m speechtotext.evaluation corpus purge-expired --manifest D:\AudioBench\aurelius-2026\manifest.json --dataset-root D:\AudioBench\aurelius-2026 --repo-root C:\src\speechtotext --report-ref-key-file D:\AudioBench\aurelius-2026\secrets\report-ref.key --receipt D:\AudioBench\aurelius-2026\reports\purge-current.journal.jsonl
```

Repite el ultimo comando con `--confirm` solo despues de revisar el primer record
`planned`. El comando reanuda exactamente ese journal; no intenta sobrescribirlo.
La purga confirmada mantiene leases sobre los mismos handles desde la
revalidacion hasta la eliminacion y flush-ea intent/outcome metadata-only. Si el
proceso cae, repite el mismo comando para reconciliar el intent pendiente antes
de continuar; no borres ni edites manualmente el journal.

## Replay

Development se usa exclusivamente para ajustar normalizacion y coeficientes del
calibrador. Calibration se usa exclusivamente para seleccionar el umbral.
Holdout se abre una sola vez, despues de congelar pipeline, modelo, features y
thresholds, y se evalua sin refit ni reseleccion.

Primero genera el artefacto requerido por Fase 2. El comando preflighta ambas
particiones antes del audit, autoriza leases JIT solo para
development+calibration y
nunca abre holdout:

Los valores `--model-id`, `AURELIUS_*_REVISION` y
`AURELIUS_*_MANIFEST_FINGERPRINT` se copian juntos desde el registry versionado
y revisado/promocionado fuera del model root. No se calculan leyendo el manifest
co-local justo antes de ejecutar, porque eso eliminaria el trust anchor. El
model root debe estar instalado read-only antes de este paso.

`train-calibrator` imprime el `artifact_fingerprint` calculado. Ese valor se
revisa y se copia a `AURELIUS_ASR_CALIBRATOR_FINGERPRINT` en la configuracion
de promocion; el replay nunca lo deriva del archivo bajo `reports/`.

```powershell
.\.venv\Scripts\python.exe -m speechtotext.evaluation train-calibrator --manifest D:\AudioBench\aurelius-2026\manifest.json --dataset-root D:\AudioBench\aurelius-2026 --repo-root C:\Users\simon\Desktop\projects\.worktrees\audio-hibrido\speechtotext --as-of 2026-07-16 --model-manifest D:\Models\fw-small\manifest.json --model-root D:\Models\fw-small --model-manifest-fingerprint $env:AURELIUS_FW_SMALL_MANIFEST_FINGERPRINT --output D:\AudioBench\aurelius-2026\reports\fw-small-es-calibrator-v1.json --artifact-version fw-small-es-v1 --min-precision-lower-95 0.99 --gain-db 0 --min-effective-voice-ms 160 --min-rms-dbfs -45 --min-snr-db 6 --max-clipping-ratio 0.01
```

Con listener, MCP y batch detenidos, promueve el artefacto al store runtime fijo.
El source permanece privado y el SHA viene del registry revisado:

```powershell
.\.venv\Scripts\python.exe -m speechtotext.security promote --source D:\AudioBench\aurelius-2026\reports\fw-small-es-calibrator-v1.json --name calibrators/fw-small-es-v1.json --expected-sha256 $env:AURELIUS_ASR_CALIBRATOR_FINGERPRINT --max-bytes 1000000
```

Los servicios posteriores mantienen `PrivateArtifactStore.runtime_session()` y
leen ese nombre con el mismo fingerprint; nunca promueven en startup/hot-reload.

```powershell
.\.venv\Scripts\python.exe -m speechtotext.evaluation --manifest D:\AudioBench\aurelius-2026\manifest.json --dataset-root D:\AudioBench\aurelius-2026 --repo-root C:\Users\simon\Desktop\projects\.worktrees\audio-hibrido\speechtotext --partition development --as-of 2026-07-16 --model-manifest D:\Models\fw-small\manifest.json --model-root D:\Models\fw-small --model-manifest-fingerprint $env:AURELIUS_FW_SMALL_MANIFEST_FINGERPRINT --model-id faster-whisper-small --model-revision $env:AURELIUS_FW_SMALL_REVISION --calibrator D:\AudioBench\aurelius-2026\reports\fw-small-es-calibrator-v1.json --calibrator-fingerprint $env:AURELIUS_ASR_CALIBRATOR_FINGERPRINT --output D:\AudioBench\aurelius-2026\reports\development.json --report-ref-key-file D:\AudioBench\aurelius-2026\secrets\report-ref.key --gain-db 0 --min-effective-voice-ms 160 --min-rms-dbfs -45 --min-snr-db 6 --max-clipping-ratio 0.01
```

El reporte contiene entorno redacted, un `job_ref` HMAC estable de los inputs
canonicos y referencias HMAC de
dataset/split/pipeline/request/modelo, quality thresholds, WER/CER con upper-95
por bloques de dia/sesion/condicion, conteo de falsos transcripts sobre silencio
sin guardar su texto, latencia interna del motor p50/p95 y upper-95 marcada
`provisional`, memoria RSS/peak y, cuando se proporciona `--calibrator`, Brier,
ECE y curva riesgo-cobertura. No contiene audio, transcripts, embeddings,
manifests de modelo, hashes crudos ni paths. `acceptance_gate.status` es
`insufficient_evidence` cuando la particion evaluada tiene menos de 30 minutos,
menos de tres dias o sesiones, o no contiene `clean`, `noise` y `silence`.
Una entrada vencida a `max(--as-of, today())` aborta antes del audit de assets,
verificacion de modelo, warm/lease y escritura de reporte.

## Comparacion local de modelos

Congela el mismo manifest, split, gain y thresholds. Ejecuta `small`,
`medium` y `large-v3-turbo` con sus manifests verificados:

```powershell
$models = @([pscustomobject]@{Name="small"; Id="faster-whisper-small"; Revision=$env:AURELIUS_FW_SMALL_REVISION; Root="D:\Models\fw-small"; Fingerprint=$env:AURELIUS_FW_SMALL_MANIFEST_FINGERPRINT}, [pscustomobject]@{Name="medium"; Id="faster-whisper-medium"; Revision=$env:AURELIUS_FW_MEDIUM_REVISION; Root="D:\Models\fw-medium"; Fingerprint=$env:AURELIUS_FW_MEDIUM_MANIFEST_FINGERPRINT}, [pscustomobject]@{Name="large-v3-turbo"; Id="faster-whisper-large-v3-turbo"; Revision=$env:AURELIUS_FW_LARGE_REVISION; Root="D:\Models\fw-large-v3-turbo"; Fingerprint=$env:AURELIUS_FW_LARGE_MANIFEST_FINGERPRINT})
foreach ($model in $models) { .\.venv\Scripts\python.exe -m speechtotext.evaluation --manifest D:\AudioBench\aurelius-2026\manifest.json --dataset-root D:\AudioBench\aurelius-2026 --repo-root C:\Users\simon\Desktop\projects\.worktrees\audio-hibrido\speechtotext --partition development --as-of 2026-07-16 --model-manifest "$($model.Root)\manifest.json" --model-root $model.Root --model-manifest-fingerprint $model.Fingerprint --model-id $model.Id --model-revision $model.Revision --output "D:\AudioBench\aurelius-2026\reports\$($model.Name)-development.json" --report-ref-key-file D:\AudioBench\aurelius-2026\secrets\report-ref.key --gain-db 0 --min-effective-voice-ms 160 --min-rms-dbfs -45 --min-snr-db 6 --max-clipping-ratio 0.01 }
```

El candidato elegible debe cumplir WER upper-95 <= 5 % limpio y <= 10 % con
ruido, cero transcript sobre silencio/noise con su upper-95 reportado y memoria
compatible con el equipo. La latencia interna del motor es diagnostica y no
aprueba 1,5 s. Entre candidatos elegibles
se selecciona el de menor WER. El holdout se ejecuta una sola vez despues de
congelar esa seleccion y el calibrador, agregando `--require-acceptance`; ese
flag sale 2 ante `failed`, `insufficient_evidence` o un reporte sin gate valido.
