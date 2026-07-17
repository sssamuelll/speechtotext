from __future__ import annotations

import hashlib
import json

import pytest

from speechtotext.models.filesystem import (
    FakeModelFilesystem,
    ModelFilesystemError,
    default_model_filesystem,
)
from speechtotext.models.manifest import (
    ModelIntegrityError,
    ModelManifest,
    VerifiedModelArtifact,
    load_model_manifest,
    parse_model_manifest_bytes,
    verify_model_files,
)


def _manifest(sha256: str) -> dict[str, object]:
    return {
        "schema_version": "speechtotext.model/v1",
        "model_id": "faster-whisper-small",
        "source": "https://huggingface.co/Systran/faster-whisper-small",
        "revision_kind": "git_commit",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "license": "MIT",
        "format": "ctranslate2",
        "sample_rate": 16000,
        "preprocessing": {"mono": True, "dtype": "float32"},
        "files": [{"path": "model.bin", "sha256": sha256}],
    }


def _fingerprint(data: dict[str, object]) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def model_fs() -> FakeModelFilesystem:
    return FakeModelFilesystem(root_read_only=True)


def _load(tmp_path, model_fs, data):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_model_manifest(
        path,
        model_root=tmp_path,
        expected_fingerprint=_fingerprint(data),
        filesystem=model_fs,
    )


def test_manifest_carga_y_fingerprint_es_estable(tmp_path, model_fs):
    (tmp_path / "model.bin").write_bytes(b"weights")
    data = _manifest(hashlib.sha256(b"weights").hexdigest())
    manifest = _load(tmp_path, model_fs, data)

    assert manifest.model_id == "faster-whisper-small"
    assert len(manifest.fingerprint) == 64
    context = verify_model_files(manifest, tmp_path, filesystem=model_fs)
    with context as artifact:
        assert artifact.manifest is manifest
        assert artifact.root == tmp_path
        assert artifact.fingerprint == manifest.fingerprint
        artifact.require_active()
        assert model_fs.leased_model_paths == {"manifest.json", "model.bin"}
    with pytest.raises(ModelIntegrityError, match="activo"):
        artifact.require_active()
    with pytest.raises(ModelIntegrityError, match="consumido"):
        with context:
            pass


def test_manifest_rechaza_campo_desconocido():
    data = _manifest("0" * 64)
    data["download_at_runtime"] = True
    with pytest.raises(ValueError, match="campos desconocidos"):
        ModelManifest.from_dict(data)


def test_integridad_falla_cerrado_si_archivo_cambio(tmp_path, model_fs):
    (tmp_path / "model.bin").write_bytes(b"tampered")
    manifest = _load(tmp_path, model_fs, _manifest("0" * 64))
    with pytest.raises(ModelIntegrityError, match="sha256"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass


@pytest.mark.parametrize(
    "unsafe",
    [
        "../secret.bin",
        "/absolute.bin",
        "sub\\model.bin",
        "model.bin:Zone.Identifier",
        "./model.bin",
        "sub//model.bin",
        "CON",
        "aux.bin",
        "model.bin.",
        "model.bin ",
        "model\x00.bin",
    ],
)
def test_manifest_rechaza_escape_o_path_windows_ambiguo(unsafe):
    data = _manifest("0" * 64)
    data["files"][0]["path"] = unsafe
    with pytest.raises(ValueError, match="ruta relativa segura"):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize("revision", ["main", "master", "latest", "v1.2.3", "01234567"])
def test_manifest_rechaza_revision_simbolica_o_abreviada(revision):
    data = _manifest("0" * 64)
    data["revision"] = revision
    with pytest.raises(ValueError, match="revision inmutable"):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize("length", [40, 64])
def test_manifest_acepta_commit_hex_completo(length):
    data = _manifest("0" * 64)
    data["revision"] = "a" * length
    manifest = ModelManifest.from_dict(data)
    assert manifest.revision == "a" * length


def test_manifest_acepta_content_digest_sha256():
    data = _manifest("0" * 64)
    data["revision_kind"] = "content_digest"
    data["revision"] = "sha256:" + "a" * 64
    assert ModelManifest.from_dict(data).revision == data["revision"]


@pytest.mark.parametrize("field", ["model_id", "source", "license", "format"])
def test_manifest_rechaza_metadata_vacia(field):
    data = _manifest("0" * 64)
    data[field] = "  "
    with pytest.raises(ValueError, match="metadata obligatoria"):
        ModelManifest.from_dict(data)


def test_manifest_desacopla_y_congela_preprocessing_anidado():
    data = _manifest("0" * 64)
    data["preprocessing"] = {"frontend": {"channels": ["mono"]}}
    manifest = ModelManifest.from_dict(data)
    fingerprint = manifest.fingerprint
    exported = manifest.to_dict()
    data["preprocessing"]["frontend"]["channels"].append("stereo")
    exported["preprocessing"]["frontend"]["channels"].append("surround")
    assert manifest.fingerprint == fingerprint
    with pytest.raises(TypeError):
        manifest.preprocessing["frontend"] = {}
    with pytest.raises(AttributeError):
        manifest.preprocessing["frontend"]["channels"].append("stereo")


def test_manifest_y_artefacto_verificado_no_tienen_constructor_publico(
    tmp_path, model_fs
):
    data = _manifest("0" * 64)
    with pytest.raises(TypeError):
        ModelManifest(**data)
    manifest = ModelManifest.from_dict(data)
    with pytest.raises(TypeError):
        VerifiedModelArtifact(manifest, tmp_path)
    with pytest.raises(TypeError, match="ModelManifest"):
        verify_model_files(object(), tmp_path, filesystem=model_fs)


def test_campos_privados_no_forjan_un_trust_anchor(tmp_path, model_fs):
    manifest = ModelManifest.from_dict(_manifest("0" * 64))
    object.__setattr__(manifest, "_trusted_fingerprint", manifest.fingerprint)
    object.__setattr__(manifest, "_manifest_relative_path", "manifest.json")
    with pytest.raises(ModelIntegrityError, match="trust anchor"):
        verify_model_files(manifest, tmp_path, filesystem=model_fs)


def test_trust_anchor_incorrecto_falla_antes_de_modelo(tmp_path, model_fs):
    data = _manifest(hashlib.sha256(b"weights").hexdigest())
    (tmp_path / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "model.bin").write_bytes(b"weights")
    with pytest.raises(ModelIntegrityError, match="trust anchor"):
        load_model_manifest(
            tmp_path / "manifest.json",
            model_root=tmp_path,
            expected_fingerprint="0" * 64,
            filesystem=model_fs,
        )
    assert "model.bin" not in model_fs.leased_model_paths


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"speechtotext.model/v1","schema_version":"x"}',
        b'{"schema_version":"speechtotext.model/v1","sample_rate":NaN}',
        b'\xff',
    ],
)
def test_manifest_rechaza_json_no_estricto(tmp_path, model_fs, payload):
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="JSON"):
        load_model_manifest(
            path,
            model_root=tmp_path,
            expected_fingerprint=hashlib.sha256(payload).hexdigest(),
            filesystem=model_fs,
        )


def test_manifest_rechaza_payload_mayor_a_un_mib():
    with pytest.raises(ValueError, match="1 MiB|payload JSON"):
        parse_model_manifest_bytes(b" " * (1_048_576 + 1))


@pytest.mark.parametrize("sample_rate", [True, 16000.5, "16000"])
def test_manifest_rechaza_coercion_de_sample_rate(sample_rate):
    data = _manifest("0" * 64)
    data["sample_rate"] = sample_rate
    with pytest.raises(ValueError, match="sample_rate"):
        ModelManifest.from_dict(data)


@pytest.mark.parametrize("expected", ["0" * 63, "A" * 64, None, True])
def test_trust_anchor_externo_exige_sha256_canonico(tmp_path, model_fs, expected):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expected_fingerprint"):
        load_model_manifest(
            tmp_path / "manifest.json",
            model_root=tmp_path,
            expected_fingerprint=expected,
            filesystem=model_fs,
        )


def test_verificacion_rechaza_extra_ausente_y_root_escribible(tmp_path, model_fs):
    (tmp_path / "model.bin").write_bytes(b"weights")
    data = _manifest(hashlib.sha256(b"weights").hexdigest())
    manifest = _load(tmp_path, model_fs, data)

    (tmp_path / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="inventario"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass
    (tmp_path / "extra.json").unlink()
    (tmp_path / "model.bin").unlink()
    with pytest.raises(ModelIntegrityError, match="inventario"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass
    (tmp_path / "model.bin").write_bytes(b"weights")
    model_fs.root_read_only = False
    with pytest.raises(ModelIntegrityError, match="read-only"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass


def test_verificacion_rechaza_directorio_extra_vacio(tmp_path, model_fs):
    (tmp_path / "model.bin").write_bytes(b"weights")
    data = _manifest(hashlib.sha256(b"weights").hexdigest())
    manifest = _load(tmp_path, model_fs, data)
    (tmp_path / "empty-dir").mkdir()
    with pytest.raises(ModelIntegrityError, match="inventario"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass


def test_modelo_activo_bloquea_replace_y_detecta_reparse(tmp_path, model_fs):
    (tmp_path / "model.bin").write_bytes(b"weights")
    data = _manifest(hashlib.sha256(b"weights").hexdigest())
    manifest = _load(tmp_path, model_fs, data)
    with verify_model_files(manifest, tmp_path, filesystem=model_fs) as artifact:
        artifact.require_active()
        for operation in (
            lambda: model_fs.replace(tmp_path / "model.bin", b"tampered"),
            lambda: model_fs.create(tmp_path / "extra.bin", b"extra"),
            lambda: model_fs.rename_root(tmp_path),
            lambda: model_fs.replace_ancestor(tmp_path.parent),
        ):
            with pytest.raises(PermissionError, match="sharing violation"):
                operation()
        sibling = tmp_path.parent / "sibling-write-ok.bin"
        model_fs.create(sibling, b"allowed outside model root")
        assert sibling.read_bytes() == b"allowed outside model root"
    model_fs.mark_reparse(tmp_path / "model.bin")
    with pytest.raises(ModelIntegrityError, match="reparse"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass


def test_verificacion_rechaza_hardlink_casefold_y_cambio_de_identidad(
    tmp_path, model_fs
):
    (tmp_path / "model.bin").write_bytes(b"weights")
    data = _manifest(hashlib.sha256(b"weights").hexdigest())
    manifest = _load(tmp_path, model_fs, data)

    model_fs.mark_hardlink(tmp_path / "model.bin")
    with pytest.raises(ModelIntegrityError, match="hardlink"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass
    model_fs.clear_faults()
    model_fs.add_inventory_alias("MODEL.BIN")
    with pytest.raises(ModelIntegrityError, match="case-fold|inventario"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass
    model_fs.clear_faults()
    model_fs.change_identity_after_lease(tmp_path / "model.bin")
    with pytest.raises(ModelIntegrityError, match="identidad"):
        with verify_model_files(manifest, tmp_path, filesystem=model_fs):
            pass


def test_default_falla_cerrado_fuera_de_windows(monkeypatch):
    monkeypatch.setattr("speechtotext.models.filesystem.sys.platform", "linux")
    with pytest.raises(ModelFilesystemError, match="adapter|Windows"):
        default_model_filesystem()
