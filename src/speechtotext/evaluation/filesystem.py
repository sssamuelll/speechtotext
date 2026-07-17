from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import sys
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO, Callable, Iterator, Protocol

if TYPE_CHECKING:
    from speechtotext.evaluation.manifest import CorpusAsset


class CorpusFilesystemError(RuntimeError):
    """El OS no pudo demostrar una operacion segura sobre el corpus."""


@dataclass(frozen=True)
class CorpusFileIdentity:
    volume_serial: int
    file_id: bytes
    size: int
    mtime_ns: int
    link_count: int


@dataclass(frozen=True)
class CorpusAssetLease:
    asset: "CorpusAsset"
    stream: BinaryIO
    identity: CorpusFileIdentity
    verified_sha256: str


@dataclass(frozen=True)
class PrivateFileLease:
    stream: BinaryIO
    identity: CorpusFileIdentity
    replaceable: bool = False


@dataclass(frozen=True)
class PrivateJournalLease:
    identity: CorpusFileIdentity
    existing_payload: bytes
    operation_id: str


class CorpusFilesystem(Protocol):
    def lease_manifest(
        self, manifest_path: Path, dataset_root: Path
    ) -> AbstractContextManager[PrivateFileLease]: ...

    def lease_manifest_for_update(
        self, manifest_path: Path, dataset_root: Path
    ) -> AbstractContextManager[PrivateFileLease]: ...

    def lease_manifest_mutation_lock(
        self, manifest_path: Path, dataset_root: Path
    ) -> AbstractContextManager[PrivateFileLease]: ...

    def atomic_replace_manifest(
        self, lease: PrivateFileLease, payload: bytes
    ) -> CorpusFileIdentity: ...

    def lease_asset(
        self, asset: "CorpusAsset", dataset_root: Path
    ) -> AbstractContextManager[CorpusAssetLease]: ...

    def delete_leased(self, lease: CorpusAssetLease) -> None: ...

    def lease_private_file(
        self, path: Path, allowed_root: Path
    ) -> AbstractContextManager[PrivateFileLease]: ...

    def atomic_write_private(
        self, path: Path, allowed_root: Path, payload: bytes
    ) -> CorpusFileIdentity: ...

    def create_private_secret(
        self, path: Path, allowed_root: Path, *, size: int
    ) -> CorpusFileIdentity: ...

    def reserve_or_resume_private_journal(
        self,
        path: Path,
        allowed_root: Path,
        initial_record: bytes,
        *,
        max_bytes: int,
    ) -> AbstractContextManager[PrivateJournalLease]: ...

    def append_private_journal(
        self, lease: PrivateJournalLease, record: bytes
    ) -> CorpusFileIdentity: ...

    def current_user_only_acl(self, path: Path) -> bool: ...

    def encryption_at_rest(self, path: Path) -> tuple[bool, str]: ...

    def identity(self, path: Path) -> CorpusFileIdentity: ...


def _abspath(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _within(path: Path, root: Path) -> bool:
    path = _abspath(path)
    root = _abspath(root)
    return path == root or root in path.parents


def _relparts(asset_path: str) -> tuple[str, ...]:
    return PurePosixPath(asset_path).parts


class FakeCorpusFilesystem:
    """Stateful fake modelando leases exclusivos, reparse points e integridad.

    No depende de Win32; simula sharing/identidad de forma suficiente para los
    tests de retencion, purga y seguridad multiplataforma.
    """

    def __init__(self) -> None:
        self.acl_ok = True
        self.encryption_ok = True
        self.provider = "fake-efs"
        self.journal_events: list[str] = []
        self._reparse_rel: set[str] = set()
        self._fail_deletes: set[str] = set()
        self._mutation_locked = False
        self._update_paths: dict[int, Path] = {}
        self._lease_paths: dict[int, Path] = {}
        self._identity_shift: dict[str, int] = {}
        self._readonly_manifest_leases = 0
        self._journal_state: dict[int, list] = {}

    # -- fault injection helpers ------------------------------------------
    def configure_security(self, *, acl_ok: bool, encryption_ok: bool) -> None:
        self.acl_ok = acl_ok
        self.encryption_ok = encryption_ok

    def replace_parent_with_reparse_point(self, rel: str, target: Path) -> None:
        del target
        self._reparse_rel.add(PurePosixPath(rel).as_posix())

    def fail_delete_for(self, rel_asset_path: str) -> None:
        self._fail_deletes.add(PurePosixPath(rel_asset_path).as_posix())

    def force_identity_change(self, path: Path) -> None:
        """Simula que la identidad OS del path cambio (mtime distinto)."""
        key = os.path.normcase(str(_abspath(path)))
        self._identity_shift[key] = self._identity_shift.get(key, 0) + 1

    # -- identity ---------------------------------------------------------
    @staticmethod
    def _raw_identity(path: Path) -> CorpusFileIdentity:
        info = path.stat(follow_symlinks=False)
        return CorpusFileIdentity(
            volume_serial=int(info.st_dev),
            file_id=int(info.st_ino).to_bytes(16, "little", signed=False),
            size=int(info.st_size),
            mtime_ns=int(info.st_mtime_ns),
            link_count=int(getattr(info, "st_nlink", 1)),
        )

    def _identity(self, path: Path) -> CorpusFileIdentity:
        identity = self._raw_identity(path)
        shift = self._identity_shift.get(os.path.normcase(str(_abspath(path))))
        if shift:
            identity = dataclasses.replace(
                identity, mtime_ns=identity.mtime_ns + shift
            )
        return identity

    def identity(self, path: Path) -> CorpusFileIdentity:
        return self._identity(_abspath(path))

    def _reject_reparse_chain(self, root: Path, rel_parts: tuple[str, ...]) -> None:
        rel = ""
        for part in rel_parts[:-1]:
            rel = f"{rel}/{part}".lstrip("/")
            if rel in self._reparse_rel:
                raise CorpusFilesystemError(f"reparse point en {rel}")

    # -- manifest leases --------------------------------------------------
    @contextmanager
    def lease_manifest(
        self, manifest_path: Path, dataset_root: Path
    ) -> Iterator[PrivateFileLease]:
        yield from self._lease_manifest(manifest_path, dataset_root, replaceable=False)

    @contextmanager
    def lease_manifest_for_update(
        self, manifest_path: Path, dataset_root: Path
    ) -> Iterator[PrivateFileLease]:
        yield from self._lease_manifest(manifest_path, dataset_root, replaceable=True)

    def _lease_manifest(
        self, manifest_path: Path, dataset_root: Path, *, replaceable: bool
    ) -> Iterator[PrivateFileLease]:
        path = _abspath(manifest_path)
        root = _abspath(dataset_root)
        if not _within(path, root):
            raise CorpusFilesystemError("manifest fuera del dataset_root")
        if not path.is_file():
            raise CorpusFilesystemError("manifest no es un archivo regular")
        # Un replay que retiene un lease read-only (sin share-delete) impide que
        # la transicion abra el lease de update: modela el bloqueo real.
        if replaceable and self._readonly_manifest_leases > 0:
            raise CorpusFilesystemError(
                "un lease read-only concurrente bloquea la transicion de mutacion"
            )
        lease = PrivateFileLease(
            stream=io.BytesIO(path.read_bytes()),
            identity=self._identity(path),
            replaceable=replaceable,
        )
        if replaceable:
            self._update_paths[id(lease)] = path
        else:
            self._readonly_manifest_leases += 1
        try:
            yield lease
        finally:
            lease.stream.close()
            self._update_paths.pop(id(lease), None)
            if not replaceable:
                self._readonly_manifest_leases -= 1

    @contextmanager
    def lease_manifest_mutation_lock(
        self, manifest_path: Path, dataset_root: Path
    ) -> Iterator[PrivateFileLease]:
        path = _abspath(manifest_path)
        if self._mutation_locked:
            raise CorpusFilesystemError("otro updater ya sostiene el lock de mutacion")
        if not path.is_file():
            raise CorpusFilesystemError("manifest no es un archivo regular")
        self._mutation_locked = True
        lock = PrivateFileLease(stream=io.BytesIO(), identity=self._identity(path))
        try:
            yield lock
        finally:
            self._mutation_locked = False

    def atomic_replace_manifest(
        self, lease: PrivateFileLease, payload: bytes
    ) -> CorpusFileIdentity:
        if not lease.replaceable:
            raise CorpusFilesystemError("lease de manifest no es replaceable")
        path = self._update_paths.get(id(lease))
        if path is None:
            raise CorpusFilesystemError("lease de update sin path asociado")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
        return self._identity(path)

    # -- asset leases -----------------------------------------------------
    @contextmanager
    def lease_asset(
        self, asset: "CorpusAsset", dataset_root: Path
    ) -> Iterator[CorpusAssetLease]:
        root = _abspath(dataset_root)
        parts = _relparts(asset.path)
        self._reject_reparse_chain(root, parts)
        path = root.joinpath(*parts)
        if not _within(path, root):
            raise CorpusFilesystemError("asset fuera del dataset_root")
        if not path.is_file():
            raise CorpusFilesystemError("asset no es un archivo regular")
        info = path.stat(follow_symlinks=False)
        if int(getattr(info, "st_nlink", 1)) != 1:
            raise CorpusFilesystemError("hardlink no permitido en asset")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != asset.sha256:
            raise CorpusFilesystemError("SHA-256 del asset no coincide")
        lease = CorpusAssetLease(
            asset=asset,
            stream=io.BytesIO(payload),
            identity=self._identity(path),
            verified_sha256=digest,
        )
        self._lease_paths[id(lease)] = path
        try:
            yield lease
        finally:
            lease.stream.close()
            self._lease_paths.pop(id(lease), None)

    def delete_leased(self, lease: CorpusAssetLease) -> None:
        self.journal_events.append("delete")
        rel = PurePosixPath(lease.asset.path).as_posix()
        if rel in self._fail_deletes:
            raise CorpusFilesystemError(f"borrado fallo para {rel}")
        path = self._lease_paths.get(id(lease))
        if path is None:
            raise CorpusFilesystemError("delete sin lease activo")
        os.remove(path)

    # -- private files / secrets ------------------------------------------
    @contextmanager
    def lease_private_file(
        self, path: Path, allowed_root: Path
    ) -> Iterator[PrivateFileLease]:
        abs_path = _abspath(path)
        if not _within(abs_path, _abspath(allowed_root)):
            raise CorpusFilesystemError("archivo privado fuera del root permitido")
        if not abs_path.is_file():
            raise CorpusFilesystemError("archivo privado no es regular")
        lease = PrivateFileLease(
            stream=io.BytesIO(abs_path.read_bytes()),
            identity=self._identity(abs_path),
        )
        try:
            yield lease
        finally:
            lease.stream.close()

    def atomic_write_private(
        self, path: Path, allowed_root: Path, payload: bytes
    ) -> CorpusFileIdentity:
        abs_path = _abspath(path)
        if not _within(abs_path, _abspath(allowed_root)):
            raise CorpusFilesystemError("destino privado fuera del root permitido")
        if abs_path.exists():
            raise CorpusFilesystemError("write privado solo acepta destino nuevo")
        tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, abs_path)
        return self._identity(abs_path)

    def create_private_secret(
        self, path: Path, allowed_root: Path, *, size: int
    ) -> CorpusFileIdentity:
        abs_path = _abspath(path)
        if not _within(abs_path, _abspath(allowed_root)):
            raise CorpusFilesystemError("secreto fuera del root permitido")
        if abs_path.exists():
            raise CorpusFilesystemError("no se reemplaza un secreto existente")
        abs_path.write_bytes(os.urandom(size))
        return self._identity(abs_path)

    # -- journal ----------------------------------------------------------
    @contextmanager
    def reserve_or_resume_private_journal(
        self,
        path: Path,
        allowed_root: Path,
        initial_record: bytes,
        *,
        max_bytes: int,
    ) -> Iterator[PrivateJournalLease]:
        abs_path = _abspath(path)
        if not _within(abs_path, _abspath(allowed_root)):
            raise CorpusFilesystemError("journal fuera del root permitido")
        if len(initial_record) > max_bytes:
            raise CorpusFilesystemError("record inicial sobre el limite")
        if abs_path.exists():
            existing = abs_path.read_bytes()
            if len(existing) > max_bytes:
                raise CorpusFilesystemError("journal existente sobre el limite")
        else:
            abs_path.write_bytes(initial_record)
            existing = initial_record
        self.journal_events.append("reserved")
        lease = PrivateJournalLease(
            identity=self._identity(abs_path),
            existing_payload=existing,
            operation_id=hashlib.sha256(initial_record).hexdigest()[:16],
        )
        # [abs_path, max_bytes, running_total] para append durable y acotado.
        self._journal_state[id(lease)] = [abs_path, max_bytes, len(existing)]
        try:
            yield lease
        finally:
            self._journal_state.pop(id(lease), None)

    def append_private_journal(
        self, lease: PrivateJournalLease, record: bytes
    ) -> CorpusFileIdentity:
        import json

        state = self._journal_state.get(id(lease))
        if state is None:
            raise CorpusFilesystemError("append sin journal activo")
        abs_path, max_bytes, total = state
        if len(record) > max_bytes or total + len(record) > max_bytes:
            raise CorpusFilesystemError("record de journal sobre el limite")
        # Append durable: el record queda en disco antes de devolver, de modo que
        # una reanudacion (reserve_or_resume) lo relee como existing_payload.
        with open(abs_path, "ab") as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        state[2] = total + len(record)
        try:
            kind = json.loads(record.decode("utf-8")).get("kind", "record")
        except (ValueError, AttributeError):
            kind = "record"
        self.journal_events.append(f"{kind}_flushed")
        return self._identity(abs_path)

    # -- security probes --------------------------------------------------
    def current_user_only_acl(self, path: Path) -> bool:
        del path
        return self.acl_ok

    def encryption_at_rest(self, path: Path) -> tuple[bool, str]:
        del path
        return self.encryption_ok, self.provider


def lease_corpus_asset(
    asset: "CorpusAsset",
    dataset_root: Path,
    *,
    filesystem: CorpusFilesystem | None = None,
) -> AbstractContextManager[CorpusAssetLease]:
    adapter = filesystem or default_corpus_filesystem()
    return adapter.lease_asset(asset, dataset_root)


# --------------------------------------------------------------------------
# Windows adapter (handle-based). Native security probes are injectable so the
# negative cases are deterministic and do not depend on CI machine config.
# --------------------------------------------------------------------------


class _WinCorpusApi:
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    DELETE = 0x00010000
    READ_CONTROL = 0x00020000
    FILE_READ_ATTRIBUTES = 0x00000080
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    CREATE_NEW = 1
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class FileTime(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        class ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", FileTime),
                ("ftLastAccessTime", FileTime),
                ("ftLastWriteTime", FileTime),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            ]

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", wintypes.BOOL)]

        self.ByHandleFileInformation = ByHandleFileInformation
        self.FileDispositionInfo = FileDispositionInfo
        self.kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self.kernel32.CreateFileW.restype = wintypes.HANDLE
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.kernel32.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ByHandleFileInformation),
        ]
        self.kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
        self.kernel32.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        self.kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        self.kernel32.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self.kernel32.SetFileInformationByHandle.restype = wintypes.BOOL

    def _error(self, code: int | None = None) -> OSError:
        return self.ctypes.WinError(
            self.ctypes.get_last_error() if code is None else code
        )

    def open(
        self,
        path: Path,
        *,
        access: int,
        share: int,
        disposition: int,
        directory: bool = False,
    ) -> int:
        flags = self.FILE_ATTRIBUTE_NORMAL | self.FILE_FLAG_OPEN_REPARSE_POINT
        if directory:
            flags |= self.FILE_FLAG_BACKUP_SEMANTICS
        handle = self.kernel32.CreateFileW(
            str(path), access, share, None, disposition, flags, None
        )
        invalid = self.ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise self._error()
        return int(handle)

    def close(self, handle: int) -> None:
        if not self.kernel32.CloseHandle(handle):
            raise self._error()

    def identity(self, handle: int) -> tuple[CorpusFileIdentity, int]:
        info = self.ByHandleFileInformation()
        if not self.kernel32.GetFileInformationByHandle(
            handle, self.ctypes.byref(info)
        ):
            raise self._error()
        file_index = (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow)
        file_time = (
            (int(info.ftLastWriteTime.dwHighDateTime) << 32)
            | int(info.ftLastWriteTime.dwLowDateTime)
        )
        identity = CorpusFileIdentity(
            volume_serial=int(info.dwVolumeSerialNumber),
            file_id=file_index.to_bytes(8, "little"),
            size=(int(info.nFileSizeHigh) << 32) | int(info.nFileSizeLow),
            mtime_ns=file_time * 100,
            link_count=int(info.nNumberOfLinks),
        )
        return identity, int(info.dwFileAttributes)

    def final_path(self, handle: int) -> Path:
        required = self.kernel32.GetFinalPathNameByHandleW(handle, None, 0, 0)
        if not required:
            raise self._error()
        buffer = self.ctypes.create_unicode_buffer(required + 1)
        written = self.kernel32.GetFinalPathNameByHandleW(
            handle, buffer, len(buffer), 0
        )
        if not written or written >= len(buffer):
            raise self._error()
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value)

    def to_stream(self, handle: int) -> BinaryIO:
        import msvcrt

        descriptor = msvcrt.open_osfhandle(handle, os.O_BINARY | os.O_RDONLY)
        return os.fdopen(descriptor, "rb", buffering=0)

    def dispose(self, handle: int) -> None:
        info = self.FileDispositionInfo(True)
        # FileDispositionInfo == 4
        if not self.kernel32.SetFileInformationByHandle(
            handle, 4, self.ctypes.byref(info), self.ctypes.sizeof(info)
        ):
            raise self._error()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


class WindowsCorpusFilesystem:
    """Adapter Windows con handles CreateFileW y probes de seguridad inyectables."""

    def __init__(
        self,
        *,
        acl_probe: Callable[[int, Path], bool] | None = None,
        encryption_probe: Callable[[int, Path], tuple[bool, str]] | None = None,
    ) -> None:
        if sys.platform != "win32":
            raise CorpusFilesystemError("el adapter Windows no existe aqui")
        self._api = _WinCorpusApi()
        self._acl_probe = acl_probe
        self._encryption_probe = encryption_probe
        self._asset_handles: dict[int, int] = {}
        self._update_target: Path | None = None
        self._journal_streams: dict[int, list] = {}

    def _verify(
        self, handle: int, path: Path, *, directory: bool, label: str
    ) -> CorpusFileIdentity:
        identity, attributes = self._api.identity(handle)
        if not _same_path(self._api.final_path(handle), path):
            raise CorpusFilesystemError(f"{label}: identidad handle-based invalida")
        if attributes & self._api.FILE_ATTRIBUTE_REPARSE_POINT:
            raise CorpusFilesystemError(f"{label}: reparse no permitido")
        is_dir = bool(attributes & self._api.FILE_ATTRIBUTE_DIRECTORY)
        if is_dir != directory:
            raise CorpusFilesystemError(f"{label}: tipo no regular")
        if not directory and identity.link_count != 1:
            raise CorpusFilesystemError(f"{label}: hardlink no permitido")
        return identity

    def _walk_reparse(self, root: Path, parts: tuple[str, ...]) -> None:
        current = root
        for part in parts[:-1]:
            current = current / part
            handle = self._api.open(
                current,
                access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
                share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_WRITE,
                disposition=self._api.OPEN_EXISTING,
                directory=True,
            )
            try:
                self._verify(handle, current, directory=True, label="componente")
            finally:
                self._api.close(handle)

    @contextmanager
    def _lease_file(
        self, path: Path, *, share: int, replaceable: bool
    ) -> Iterator[PrivateFileLease]:
        handle = self._api.open(
            path,
            access=self._api.GENERIC_READ | self._api.FILE_READ_ATTRIBUTES,
            share=share,
            disposition=self._api.OPEN_EXISTING,
        )
        stream: BinaryIO | None = None
        try:
            identity = self._verify(handle, path, directory=False, label="manifest")
            stream = self._api.to_stream(handle)
            handle = -1
            yield PrivateFileLease(stream, identity, replaceable=replaceable)
        finally:
            if stream is not None:
                stream.close()
            elif handle != -1:
                self._api.close(handle)

    @contextmanager
    def lease_manifest(
        self, manifest_path: Path, dataset_root: Path
    ) -> Iterator[PrivateFileLease]:
        del dataset_root
        with self._lease_file(
            Path(manifest_path),
            share=self._api.FILE_SHARE_READ,
            replaceable=False,
        ) as lease:
            yield lease

    @contextmanager
    def lease_manifest_for_update(
        self, manifest_path: Path, dataset_root: Path
    ) -> Iterator[PrivateFileLease]:
        del dataset_root
        with self._lease_file(
            Path(manifest_path),
            share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_DELETE,
            replaceable=True,
        ) as lease:
            self._update_target = Path(manifest_path)
            yield lease

    @contextmanager
    def lease_manifest_mutation_lock(
        self, manifest_path: Path, dataset_root: Path
    ) -> Iterator[PrivateFileLease]:
        del dataset_root
        handle = self._api.open(
            Path(manifest_path),
            access=self._api.FILE_READ_ATTRIBUTES,
            share=self._api.FILE_SHARE_READ,
            disposition=self._api.OPEN_EXISTING,
        )
        try:
            identity = self._verify(
                handle, Path(manifest_path), directory=False, label="lock"
            )
            yield PrivateFileLease(io.BytesIO(), identity)
        finally:
            self._api.close(handle)

    def atomic_replace_manifest(
        self, lease: PrivateFileLease, payload: bytes
    ) -> CorpusFileIdentity:
        if not lease.replaceable:
            raise CorpusFilesystemError("lease de manifest no es replaceable")
        target = getattr(self, "_update_target", None)
        if target is None:
            raise CorpusFilesystemError("lease de update sin path asociado")
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, target)
        return self.identity(target)

    @contextmanager
    def lease_asset(
        self, asset: "CorpusAsset", dataset_root: Path
    ) -> Iterator[CorpusAssetLease]:
        import msvcrt

        root = _abspath(Path(dataset_root))
        parts = _relparts(asset.path)
        self._walk_reparse(root, parts)
        path = root.joinpath(*parts)
        handle = self._api.open(
            path,
            access=self._api.GENERIC_READ
            | self._api.DELETE
            | self._api.FILE_READ_ATTRIBUTES,
            share=self._api.FILE_SHARE_READ,
            disposition=self._api.OPEN_EXISTING,
        )
        stream: BinaryIO | None = None
        try:
            identity = self._verify(handle, path, directory=False, label="asset")
            descriptor = msvcrt.open_osfhandle(handle, os.O_BINARY | os.O_RDONLY)
            stream = os.fdopen(descriptor, "rb", buffering=0)
            hasher = hashlib.sha256()
            for block in iter(lambda: stream.read(65536), b""):
                hasher.update(block)
            digest = hasher.hexdigest()
            if digest != asset.sha256:
                raise CorpusFilesystemError("SHA-256 del asset no coincide")
            stream.seek(0)
            lease = CorpusAssetLease(asset, stream, identity, digest)
            self._asset_handles[id(lease)] = handle
            handle = -1
            try:
                yield lease
            finally:
                self._asset_handles.pop(id(lease), None)
        finally:
            if stream is not None:
                stream.close()
            elif handle != -1:
                self._api.close(handle)

    def delete_leased(self, lease: CorpusAssetLease) -> None:
        # Delete-on-close via the same leased handle; never stat-then-unlink.
        handle = self._asset_handles.get(id(lease))
        if handle is None:
            raise CorpusFilesystemError("delete sin lease activo")
        self._api.dispose(handle)

    @contextmanager
    def lease_private_file(
        self, path: Path, allowed_root: Path
    ) -> Iterator[PrivateFileLease]:
        if not _within(Path(path), Path(allowed_root)):
            raise CorpusFilesystemError("archivo privado fuera del root permitido")
        with self._lease_file(
            Path(path), share=self._api.FILE_SHARE_READ, replaceable=False
        ) as lease:
            yield lease

    def _create_new_and_write(self, target: Path, payload: bytes) -> None:
        import msvcrt

        handle = self._api.open(
            target,
            access=self._api.GENERIC_WRITE,
            share=self._api.FILE_SHARE_READ,
            disposition=self._api.CREATE_NEW,
        )
        try:
            self._require_secure(handle, target)
        except BaseException:
            self._api.close(handle)
            raise
        with os.fdopen(
            msvcrt.open_osfhandle(handle, os.O_BINARY), "wb", buffering=0, closefd=True
        ) as writer:
            writer.write(payload)
            writer.flush()

    def atomic_write_private(
        self, path: Path, allowed_root: Path, payload: bytes
    ) -> CorpusFileIdentity:
        if not _within(Path(path), Path(allowed_root)):
            raise CorpusFilesystemError("destino privado fuera del root permitido")
        target = _abspath(Path(path))
        if target.exists():
            raise CorpusFilesystemError("write privado solo acepta destino nuevo")
        tmp = target.with_suffix(target.suffix + ".tmp")
        if tmp.exists():
            os.remove(tmp)
        self._create_new_and_write(tmp, payload)
        os.replace(tmp, target)
        return self.identity(target)

    def create_private_secret(
        self, path: Path, allowed_root: Path, *, size: int
    ) -> CorpusFileIdentity:
        if not _within(Path(path), Path(allowed_root)):
            raise CorpusFilesystemError("secreto fuera del root permitido")
        target = _abspath(Path(path))
        if target.exists():
            raise CorpusFilesystemError("no se reemplaza un secreto existente")
        self._create_new_and_write(target, os.urandom(size))
        return self.identity(target)

    @contextmanager
    def reserve_or_resume_private_journal(
        self,
        path: Path,
        allowed_root: Path,
        initial_record: bytes,
        *,
        max_bytes: int,
    ) -> Iterator[PrivateJournalLease]:
        import msvcrt

        if not _within(Path(path), Path(allowed_root)):
            raise CorpusFilesystemError("journal fuera del root permitido")
        target = _abspath(Path(path))
        if len(initial_record) > max_bytes:
            raise CorpusFilesystemError("record inicial sobre el limite")
        resuming = target.exists()
        if resuming:
            existing = target.read_bytes()
            if len(existing) > max_bytes:
                raise CorpusFilesystemError("journal existente sobre el limite")
            disposition = self._api.OPEN_EXISTING
        else:
            existing = initial_record
            disposition = self._api.CREATE_NEW
        handle = self._api.open(
            target,
            access=self._api.GENERIC_WRITE | self._api.FILE_READ_ATTRIBUTES,
            share=self._api.FILE_SHARE_READ,
            disposition=disposition,
        )
        stream: BinaryIO | None = None
        try:
            self._require_secure(handle, target)
            stream = os.fdopen(
                msvcrt.open_osfhandle(handle, os.O_BINARY | os.O_APPEND),
                "wb",
                buffering=0,
                closefd=True,
            )
            handle = -1
            if not resuming:
                stream.write(initial_record)
                stream.flush()
                os.fsync(stream.fileno())
            lease = PrivateJournalLease(
                identity=self.identity(target),
                existing_payload=existing,
                operation_id=hashlib.sha256(initial_record).hexdigest()[:16],
            )
            self._journal_streams[id(lease)] = [stream, target, max_bytes, len(existing)]
            try:
                yield lease
            finally:
                self._journal_streams.pop(id(lease), None)
        finally:
            if stream is not None:
                stream.close()
            elif handle != -1:
                self._api.close(handle)

    def append_private_journal(
        self, lease: PrivateJournalLease, record: bytes
    ) -> CorpusFileIdentity:
        state = self._journal_streams.get(id(lease))
        if state is None:
            raise CorpusFilesystemError("append sin journal activo")
        stream, target, max_bytes, total = state
        if len(record) > max_bytes or total + len(record) > max_bytes:
            raise CorpusFilesystemError("record de journal sobre el limite")
        # Append durable sobre el mismo handle: flush + FlushFileBuffers (fsync)
        # antes de devolver, para que el intent preceda al side effect destructivo.
        stream.write(record)
        stream.flush()
        os.fsync(stream.fileno())
        state[3] = total + len(record)
        return self.identity(target)

    def _require_secure(self, handle: int, path: Path) -> None:
        if self._acl_probe is not None and not self._acl_probe(handle, path):
            raise CorpusFilesystemError("DACL current-user-only no demostrada")
        if self._encryption_probe is not None:
            ok, _provider = self._encryption_probe(handle, path)
            if not ok:
                raise CorpusFilesystemError("cifrado en reposo no demostrado")

    def current_user_only_acl(self, path: Path) -> bool:
        handle = self._api.open(
            Path(path),
            access=self._api.READ_CONTROL | self._api.FILE_READ_ATTRIBUTES,
            share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_WRITE,
            disposition=self._api.OPEN_EXISTING,
            directory=Path(path).is_dir(),
        )
        try:
            if self._acl_probe is None:
                return False
            return bool(self._acl_probe(handle, Path(path)))
        finally:
            self._api.close(handle)

    def encryption_at_rest(self, path: Path) -> tuple[bool, str]:
        handle = self._api.open(
            Path(path),
            access=self._api.READ_CONTROL | self._api.FILE_READ_ATTRIBUTES,
            share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_WRITE,
            disposition=self._api.OPEN_EXISTING,
            directory=Path(path).is_dir(),
        )
        try:
            if self._encryption_probe is None:
                return False, "unproven"
            return self._encryption_probe(handle, Path(path))
        finally:
            self._api.close(handle)

    def identity(self, path: Path) -> CorpusFileIdentity:
        directory = Path(path).is_dir()
        handle = self._api.open(
            Path(path),
            access=self._api.FILE_READ_ATTRIBUTES,
            share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_WRITE,
            disposition=self._api.OPEN_EXISTING,
            directory=directory,
        )
        try:
            identity, _attrs = self._api.identity(handle)
            return identity
        finally:
            self._api.close(handle)


def default_corpus_filesystem() -> CorpusFilesystem:
    if sys.platform != "win32":
        raise CorpusFilesystemError(
            "se requiere un adapter verificable; solo Windows esta disponible"
        )
    return WindowsCorpusFilesystem()
