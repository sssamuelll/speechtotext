from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import io
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import BinaryIO, Protocol


MAX_PRIVATE_ARTIFACT_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_PRIVATE_TEMP_RE = re.compile(r"\.artifact-[0-9a-f]{32}\.tmp", re.ASCII)
_WIN_FORBIDDEN = frozenset('<>:"\\|?*')
_WIN_RESERVED = frozenset({"con", "prn", "aux", "nul"})
_WIN_DEVICE_SUFFIXES = frozenset((*"123456789", "\u00b9", "\u00b2", "\u00b3"))
_FACTORY_TOKEN = object()


class ArtifactIntegrityError(RuntimeError):
    """Identity, privacy, or integrity could not be demonstrated."""


@dataclass(frozen=True)
class ArtifactPathIdentity:
    volume_serial: int
    file_id: bytes
    size: int
    mtime_ns: int
    link_count: int


@dataclass
class _LeaseState:
    active: bool = True


@dataclass(frozen=True)
class ArtifactRootLease:
    root: Path
    path_chain: tuple[ArtifactPathIdentity, ...]
    encryption_provider: str
    _state: _LeaseState = field(default_factory=_LeaseState, repr=False, compare=False)
    _handles: tuple[object, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class ArtifactFileHandle:
    relative_name: str
    stream: BinaryIO
    identity: ArtifactPathIdentity
    encryption_provider: str


@dataclass(frozen=True)
class ArtifactSourceHandle:
    stream: BinaryIO
    identity: ArtifactPathIdentity
    encryption_provider: str


class PrivateArtifactFilesystem(Protocol):
    def lease_current_user_root(
        self,
    ) -> AbstractContextManager[ArtifactRootLease]: ...

    def lease_file(
        self, relative_name: str, root: ArtifactRootLease
    ) -> AbstractContextManager[ArtifactFileHandle]: ...

    def runtime_session(self) -> AbstractContextManager[None]: ...

    def offline_promotion(self) -> AbstractContextManager[None]: ...

    def lease_private_source(
        self, source: Path
    ) -> AbstractContextManager[ArtifactSourceHandle]: ...

    def install_create_new(
        self,
        relative_name: str,
        root: ArtifactRootLease,
        source: ArtifactSourceHandle,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> ArtifactPathIdentity: ...


def _safe_relative(value: str, *, allow_dot_prefix: bool = False) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("el artefacto exige una ruta relativa segura")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("el artefacto exige una ruta relativa segura")
    for part in path.parts:
        stem = part.split(".", 1)[0].casefold()
        device = stem in _WIN_RESERVED or (
            len(stem) == 4
            and stem[:3] in {"com", "lpt"}
            and stem[3] in _WIN_DEVICE_SUFFIXES
        )
        if (
            # Leading-dot components are reserved for store internals
            # (.artifact-*.tmp temps and .runtime.lock); rejecting them all
            # keeps promotion from publishing names the cleanup would delete.
            # The rule applies to store-relative DESTINATION names only:
            # promotion source ancestors (allow_dot_prefix) may live under
            # dot-directories like .cache.
            (not allow_dot_prefix and part.startswith("."))
            or part.endswith((".", " "))
            or any(ord(char) < 32 or char in _WIN_FORBIDDEN for char in part)
            or device
            or len(part.encode("utf-16-le")) // 2 > 255
        ):
            raise ValueError("el artefacto exige una ruta relativa segura")
    if len(value.encode("utf-16-le")) // 2 > 32767:
        raise ValueError("el artefacto exige una ruta relativa segura")
    return value


def _validate_sha256(value: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("expected_sha256 exige SHA-256 lowercase canonico")
    return value


def _validate_max_bytes(value: int) -> int:
    if type(value) is not int or value <= 0 or value > MAX_PRIVATE_ARTIFACT_BYTES:
        raise ValueError(f"max_bytes debe estar entre 1 y {MAX_PRIVATE_ARTIFACT_BYTES}")
    return value


def _read_bounded(stream: BinaryIO, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = max_bytes - total
        chunk = stream.read(min(_READ_CHUNK_BYTES, remaining + 1))
        if not isinstance(chunk, bytes):
            raise ArtifactIntegrityError("la lectura del artefacto no devolvio bytes")
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ArtifactIntegrityError("el artefacto excede el limite permitido")
        chunks.append(chunk)
    return b"".join(chunks)


def _copy_bounded_with_sha256(
    source: BinaryIO, destination: BinaryIO, max_bytes: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while True:
        remaining = max_bytes - total
        chunk = source.read(min(_READ_CHUNK_BYTES, remaining + 1))
        if not isinstance(chunk, bytes):
            raise ArtifactIntegrityError("la lectura del source no devolvio bytes")
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ArtifactIntegrityError("el source excede el limite permitido")
        if destination.write(chunk) != len(chunk):
            raise ArtifactIntegrityError("copia parcial de artefacto privado")
        digest.update(chunk)
    return total, digest.hexdigest()


def _require_secure_file(
    identity: ArtifactPathIdentity, encryption_provider: str
) -> None:
    if identity.link_count != 1:
        raise ArtifactIntegrityError("hardlink no permitido en artefacto privado")
    if identity.size < 0:
        raise ArtifactIntegrityError("identidad de artefacto invalida")
    if not encryption_provider:
        raise ArtifactIntegrityError("encryption no demostrado")


class ArtifactLease(AbstractContextManager["ArtifactLease"]):
    """Sealed, single-consumption lease returned by PrivateArtifactStore."""

    __slots__ = (
        "_active",
        "_consumed",
        "_expected_sha256",
        "_handle",
        "_max_bytes",
    )

    def __init__(
        self,
        *,
        _token: object,
        handle: ArtifactFileHandle,
        expected_sha256: str,
        max_bytes: int,
    ) -> None:
        if _token is not _FACTORY_TOKEN:
            raise TypeError("ArtifactLease solo se obtiene desde PrivateArtifactStore")
        self._handle = handle
        self._expected_sha256 = expected_sha256
        self._max_bytes = max_bytes
        self._active = True
        self._consumed = False

    def __enter__(self) -> ArtifactLease:
        self.require_active()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._active = False

    def require_active(self) -> None:
        if not self._active:
            raise ArtifactIntegrityError("el lease no esta activo")

    def read_bytes_once(self) -> bytes:
        self.require_active()
        if self._consumed:
            raise ArtifactIntegrityError("el lease ya fue consumido")
        self._consumed = True
        if self._handle.identity.size > self._max_bytes:
            raise ArtifactIntegrityError("el artefacto excede el limite permitido")
        payload = _read_bounded(self._handle.stream, self._max_bytes)
        if len(payload) != self._handle.identity.size:
            raise ArtifactIntegrityError(
                "el tamano leido no coincide con la identidad del handle"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(digest, self._expected_sha256):
            raise ArtifactIntegrityError("sha256 del artefacto no coincide")
        return payload


class _ArtifactLeaseContext(AbstractContextManager[ArtifactLease]):
    def __init__(
        self,
        filesystem: PrivateArtifactFilesystem,
        relative_name: str,
        expected_sha256: str,
        max_bytes: int,
    ) -> None:
        self._filesystem = filesystem
        self._relative_name = relative_name
        self._expected_sha256 = expected_sha256
        self._max_bytes = max_bytes
        self._stack: ExitStack | None = None
        self._lease: ArtifactLease | None = None

    def __enter__(self) -> ArtifactLease:
        if self._stack is not None:
            raise ArtifactIntegrityError("el context manager de lease ya fue usado")
        stack = ExitStack()
        self._stack = stack
        try:
            root = stack.enter_context(self._filesystem.lease_current_user_root())
            handle = stack.enter_context(
                self._filesystem.lease_file(self._relative_name, root)
            )
            _require_secure_file(handle.identity, handle.encryption_provider)
            lease = ArtifactLease(
                _token=_FACTORY_TOKEN,
                handle=handle,
                expected_sha256=self._expected_sha256,
                max_bytes=self._max_bytes,
            )
            self._lease = lease
            return lease
        except BaseException:
            stack.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._lease is not None:
            self._lease.__exit__(exc_type, exc_value, traceback)
        if self._stack is not None:
            self._stack.close()


class PrivateArtifactStore:
    __slots__ = ("_filesystem",)

    def __init__(
        self,
        *,
        _token: object,
        filesystem: PrivateArtifactFilesystem,
    ) -> None:
        if _token is not _FACTORY_TOKEN:
            raise TypeError("PrivateArtifactStore solo se obtiene desde current_user")
        self._filesystem = filesystem

    @classmethod
    def current_user(
        cls, *, filesystem: PrivateArtifactFilesystem | None = None
    ) -> PrivateArtifactStore:
        selected = (
            filesystem
            if filesystem is not None
            else default_private_artifact_filesystem()
        )
        return cls(_token=_FACTORY_TOKEN, filesystem=selected)

    def lease(
        self,
        relative_name: str,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> AbstractContextManager[ArtifactLease]:
        safe_name = _safe_relative(relative_name)
        expected = _validate_sha256(expected_sha256)
        limit = _validate_max_bytes(max_bytes)
        return _ArtifactLeaseContext(self._filesystem, safe_name, expected, limit)

    def runtime_session(self) -> AbstractContextManager[None]:
        return self._filesystem.runtime_session()

    def promote_from_path(
        self,
        source: Path,
        relative_name: str,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> None:
        if not isinstance(source, Path):
            source = Path(source)
        safe_name = _safe_relative(relative_name)
        expected = _validate_sha256(expected_sha256)
        limit = _validate_max_bytes(max_bytes)
        with self._filesystem.offline_promotion():
            with self._filesystem.lease_current_user_root() as root:
                with self._filesystem.lease_private_source(source) as source_handle:
                    _require_secure_file(
                        source_handle.identity,
                        source_handle.encryption_provider,
                    )
                    self._filesystem.install_create_new(
                        safe_name,
                        root,
                        source_handle,
                        expected_sha256=expected,
                        max_bytes=limit,
                    )


@dataclass
class _FakeArtifact:
    payload: bytes
    generation: int
    faults: set[str] = field(default_factory=set)


class _FakeStream(io.BytesIO):
    def __init__(self, payload: bytes, faults: set[str]) -> None:
        if "grow_after_stat" in faults:
            payload += b"5"
        super().__init__(payload)
        self._faults = faults

    def read(self, size: int = -1) -> bytes:
        if "read" in self._faults:
            raise OSError("fallo de lectura inyectado")
        return super().read(size)


class FakePrivateArtifactFilesystem:
    """Stateful fake that models root, file, and promotion leases."""

    def __init__(
        self,
        *,
        known_local_app_data: Path,
        acl_ok: bool,
        encryption_ok: bool,
    ) -> None:
        self.known_local_app_data = Path(known_local_app_data)
        self.root = self.known_local_app_data / "speechtotext" / "artifacts"
        self._acl_ok = acl_ok
        self._encryption_ok = encryption_ok
        self._artifacts: dict[str, _FakeArtifact] = {}
        self._sources: dict[Path, _FakeArtifact] = {}
        self._source_faults: dict[Path, set[str]] = {}
        self._root_faults: dict[str, set[str]] = {}
        self._leased_names: Counter[str] = Counter()
        self._generation = 0
        self._runtime_sessions = 0
        self._promotion_active = False
        self._promotion_fault: str | None = None
        self._temps: dict[str, bytes] = {}
        self.open_counts: Counter[str] = Counter()
        self.source_open_count = 0
        self.promotion_events: list[str] = []

    @property
    def total_open_count(self) -> int:
        return sum(self.open_counts.values())

    @property
    def temp_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._temps))

    def _next_identity(
        self, name: str, artifact: _FakeArtifact
    ) -> ArtifactPathIdentity:
        links = 2 if "hardlink" in artifact.faults else 1
        file_id = hashlib.sha256(
            f"{name}:{artifact.generation}".encode("utf-8")
        ).digest()[:16]
        return ArtifactPathIdentity(
            volume_serial=1,
            file_id=file_id,
            size=len(artifact.payload),
            mtime_ns=artifact.generation,
            link_count=links,
        )

    def install(self, relative_name: str, payload: bytes) -> None:
        name = _safe_relative(relative_name)
        self._generation += 1
        self._artifacts[name] = _FakeArtifact(bytes(payload), self._generation)

    def read(self, relative_name: str) -> bytes:
        return self._artifacts[relative_name].payload

    def exists(self, relative_name: str) -> bool:
        return relative_name in self._artifacts

    def replace(self, relative_name: str, payload: bytes) -> None:
        if self._leased_names[relative_name]:
            raise PermissionError("sharing violation")
        self.install(relative_name, payload)

    def inject_fault(self, relative_name: str, fault: str) -> None:
        self._artifacts[relative_name].faults.add(fault)

    def inject_root_fault(self, component: str, fault: str) -> None:
        self._root_faults.setdefault(component, set()).add(fault)

    def private_source(self, relative_name: str, payload: bytes) -> Path:
        name = _safe_relative(relative_name)
        source = self.known_local_app_data / "private-sources"
        source = source.joinpath(*PurePosixPath(name).parts)
        self._generation += 1
        self._sources[source] = _FakeArtifact(bytes(payload), self._generation)
        return source

    def inject_source_fault(self, source: Path, fault: str) -> None:
        self._source_faults.setdefault(Path(source), set()).add(fault)

    def clear_source_faults(self, source: Path) -> None:
        self._source_faults.pop(Path(source), None)

    def inject_promotion_fault(self, fault: str) -> None:
        self._promotion_fault = fault

    def install_temp(self, name: str, payload: bytes) -> None:
        self._temps[name] = bytes(payload)

    def clear_promotion_events(self) -> None:
        self.promotion_events.clear()

    def _raise_root_faults(self) -> None:
        if not self._acl_ok:
            raise ArtifactIntegrityError("artifacts root acl no demostrada")
        if not self._encryption_ok:
            raise ArtifactIntegrityError("artifacts root encryption no demostrada")
        for component in (
            "known_local_app_data",
            "speechtotext",
            "artifacts",
        ):
            for fault in sorted(self._root_faults.get(component, ())):
                raise ArtifactIntegrityError(f"{component}: {fault} no demostrado")

    @contextmanager
    def lease_current_user_root(self) -> Iterator[ArtifactRootLease]:
        self._raise_root_faults()
        identities = tuple(
            ArtifactPathIdentity(
                volume_serial=1,
                file_id=hashlib.sha256(component.encode()).digest()[:16],
                size=0,
                mtime_ns=1,
                link_count=1,
            )
            for component in (
                "known_local_app_data",
                "speechtotext",
                "artifacts",
            )
        )
        lease = ArtifactRootLease(
            root=self.root,
            path_chain=identities,
            encryption_provider="fake-encryption",
        )
        try:
            yield lease
        finally:
            lease._state.active = False

    @contextmanager
    def lease_file(
        self, relative_name: str, root: ArtifactRootLease
    ) -> Iterator[ArtifactFileHandle]:
        if not root._state.active or root.root != self.root:
            raise ArtifactIntegrityError("root lease no esta activo")
        artifact = self._artifacts.get(relative_name)
        if artifact is None:
            raise ArtifactIntegrityError("artefacto privado no encontrado")
        self.open_counts[relative_name] += 1
        for fault in sorted(artifact.faults):
            if fault in {"acl", "encryption", "reparse"}:
                raise ArtifactIntegrityError(f"{fault} no demostrado")
        stream = _FakeStream(artifact.payload, artifact.faults)
        self._leased_names[relative_name] += 1
        handle = ArtifactFileHandle(
            relative_name=relative_name,
            stream=stream,
            identity=self._next_identity(relative_name, artifact),
            encryption_provider=(
                "fake-encryption" if "encryption" not in artifact.faults else ""
            ),
        )
        try:
            yield handle
        finally:
            stream.close()
            self._leased_names[relative_name] -= 1

    @contextmanager
    def runtime_session(self) -> Iterator[None]:
        if self._promotion_active:
            raise ArtifactIntegrityError("promocion offline activa")
        self._runtime_sessions += 1
        try:
            yield None
        finally:
            self._runtime_sessions -= 1

    @contextmanager
    def offline_promotion(self) -> Iterator[None]:
        if self._runtime_sessions or self._promotion_active:
            raise ArtifactIntegrityError("servicio activo impide promocion")
        self._promotion_active = True
        self.promotion_events.append("offline_lock")
        try:
            for name in tuple(self._temps):
                if _PRIVATE_TEMP_RE.fullmatch(name):
                    self._temps.pop(name)
            yield None
        finally:
            self._promotion_active = False

    @contextmanager
    def lease_private_source(self, source: Path) -> Iterator[ArtifactSourceHandle]:
        if not self._promotion_active:
            raise ArtifactIntegrityError("source solo se abre bajo promocion offline")
        path = Path(source)
        artifact = self._sources.get(path)
        if artifact is None:
            raise ArtifactIntegrityError("source privado no encontrado")
        self.source_open_count += 1
        self.promotion_events.append("source_leased")
        faults = set(self._source_faults.get(path, ()))
        for fault in sorted(faults):
            if fault.startswith("exception:"):
                raise RuntimeError(fault.removeprefix("exception:"))
            if fault in {"acl", "encryption", "reparse"}:
                raise ArtifactIntegrityError(f"source {fault} no demostrado")
        effective_faults = set(faults)
        if "tamper" in faults:
            effective_faults.add("grow_after_stat")
        identity_artifact = _FakeArtifact(
            artifact.payload, artifact.generation, effective_faults
        )
        stream = _FakeStream(artifact.payload, effective_faults)
        handle = ArtifactSourceHandle(
            stream=stream,
            identity=self._next_identity(str(path), identity_artifact),
            encryption_provider=(
                "fake-encryption" if "encryption" not in faults else ""
            ),
        )
        try:
            yield handle
        finally:
            stream.close()

    def install_create_new(
        self,
        relative_name: str,
        root: ArtifactRootLease,
        source: ArtifactSourceHandle,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> ArtifactPathIdentity:
        if not self._promotion_active or not root._state.active:
            raise ArtifactIntegrityError("promocion offline no esta activa")
        if relative_name in self._artifacts:
            raise ArtifactIntegrityError("colision con artefacto existente")
        temp_name = f".artifact-{self._generation + 1}.tmp"
        self._temps[temp_name] = b""
        self.promotion_events.append("temp_secured")
        committed = False
        try:
            payload = _read_bounded(source.stream, max_bytes)
            if not hmac.compare_digest(
                hashlib.sha256(payload).hexdigest(), expected_sha256
            ):
                raise ArtifactIntegrityError("sha256 de source no coincide")
            self._temps[temp_name] = payload
            self.promotion_events.append("temp_flushed")
            if self._promotion_fault == "crash_before_commit":
                raise ArtifactIntegrityError("promocion interrumpida antes de commit")
            if relative_name in self._artifacts:
                raise ArtifactIntegrityError("colision con artefacto existente")
            self._generation += 1
            self._artifacts[relative_name] = _FakeArtifact(payload, self._generation)
            committed = True
            self._temps.pop(temp_name, None)
            self.promotion_events.append("create_new_committed")
            self.promotion_events.append("destination_reopened")
            if self._promotion_fault == "destination_tamper":
                raise ArtifactIntegrityError("destino de promocion no pudo revalidarse")
            destination = self._artifacts[relative_name]
            identity = self._next_identity(relative_name, destination)
            if not hmac.compare_digest(
                hashlib.sha256(destination.payload).hexdigest(), expected_sha256
            ):
                raise ArtifactIntegrityError("destino de promocion no coincide")
            self.promotion_events.append("destination_verified")
            return identity
        except BaseException:
            self._temps.pop(temp_name, None)
            if committed:
                self._artifacts.pop(relative_name, None)
            raise
        finally:
            self._promotion_fault = None


class _Win32Api:
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    DELETE = 0x00010000
    READ_CONTROL = 0x00020000
    WRITE_DAC = 0x00040000
    FILE_READ_ATTRIBUTES = 0x00000080
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    CREATE_NEW = 1
    OPEN_EXISTING = 3
    OPEN_ALWAYS = 4
    FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    FILE_ATTRIBUTE_ENCRYPTED = 0x00004000
    FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    MOVEFILE_WRITE_THROUGH = 0x00000008
    LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    ERROR_FILE_EXISTS = 80
    ERROR_ALREADY_EXISTS = 183
    ERROR_LOCK_VIOLATION = 33
    ERROR_SHARING_VIOLATION = 32

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

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

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]

        class AclSizeInformation(ctypes.Structure):
            _fields_ = [
                ("AceCount", wintypes.DWORD),
                ("AclBytesInUse", wintypes.DWORD),
                ("AclBytesFree", wintypes.DWORD),
            ]

        class AceHeader(ctypes.Structure):
            _fields_ = [
                ("AceType", ctypes.c_ubyte),
                ("AceFlags", ctypes.c_ubyte),
                ("AceSize", wintypes.WORD),
            ]

        class AccessAllowedAce(ctypes.Structure):
            _fields_ = [
                ("Header", AceHeader),
                ("Mask", wintypes.DWORD),
                ("SidStart", wintypes.DWORD),
            ]

        class SidAndAttributes(ctypes.Structure):
            _fields_ = [
                ("Sid", wintypes.LPVOID),
                ("Attributes", wintypes.DWORD),
            ]

        class TokenUser(ctypes.Structure):
            _fields_ = [("User", SidAndAttributes)]

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", wintypes.BOOL)]

        self.FileTime = FileTime
        self.ByHandleFileInformation = ByHandleFileInformation
        self.Overlapped = Overlapped
        self.AclSizeInformation = AclSizeInformation
        self.AceHeader = AceHeader
        self.AccessAllowedAce = AccessAllowedAce
        self.TokenUser = TokenUser
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
        self.kernel32.CreateDirectoryW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPVOID,
        ]
        self.kernel32.CreateDirectoryW.restype = wintypes.BOOL
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
        self.kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        self.kernel32.FlushFileBuffers.restype = wintypes.BOOL
        self.kernel32.MoveFileExW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
        ]
        self.kernel32.MoveFileExW.restype = wintypes.BOOL
        self.kernel32.LockFileEx.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(Overlapped),
        ]
        self.kernel32.LockFileEx.restype = wintypes.BOOL
        self.kernel32.UnlockFileEx.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(Overlapped),
        ]
        self.kernel32.UnlockFileEx.restype = wintypes.BOOL
        self.kernel32.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self.kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
        self.kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        self.advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.advapi32.OpenProcessToken.restype = wintypes.BOOL
        self.advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.GetTokenInformation.restype = wintypes.BOOL
        self.advapi32.GetLengthSid.argtypes = [wintypes.LPVOID]
        self.advapi32.GetLengthSid.restype = wintypes.DWORD
        self.advapi32.GetSecurityInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.LPVOID),
        ]
        self.advapi32.GetSecurityInfo.restype = wintypes.DWORD
        self.advapi32.GetAclInformation.argtypes = [
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.c_int,
        ]
        self.advapi32.GetAclInformation.restype = wintypes.BOOL
        self.advapi32.GetAce.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
        ]
        self.advapi32.GetAce.restype = wintypes.BOOL
        self.advapi32.ConvertSidToStringSidW.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        self.advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
        self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
            wintypes.BOOL
        )
        self.advapi32.EncryptFileW.argtypes = [wintypes.LPCWSTR]
        self.advapi32.EncryptFileW.restype = wintypes.BOOL
        self.advapi32.GetSecurityDescriptorDacl.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.BOOL),
        ]
        self.advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
        self.advapi32.GetSecurityDescriptorControl.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
        self.advapi32.SetSecurityInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        self.advapi32.SetSecurityInfo.restype = wintypes.DWORD
        self.kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
        self.kernel32.LocalFree.restype = wintypes.HLOCAL

    def _win_error(self, code: int | None = None) -> OSError:
        if code is None:
            code = self.ctypes.get_last_error()
        return self.ctypes.WinError(code)

    def open(
        self,
        path: Path,
        *,
        access: int,
        share: int,
        creation: int,
        directory: bool = False,
    ) -> int:
        flags = self.FILE_FLAG_OPEN_REPARSE_POINT
        flags |= self.FILE_FLAG_BACKUP_SEMANTICS if directory else 0
        flags |= self.FILE_ATTRIBUTE_NORMAL
        handle = self.kernel32.CreateFileW(
            str(path), access, share, None, creation, flags, None
        )
        invalid = self.ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise self._win_error()
        return int(handle)

    def create_directory(self, path: Path) -> bool:
        """CREATE_NEW semantics: True if created, False if it already existed."""
        if self.kernel32.CreateDirectoryW(str(path), None):
            return True
        code = self.ctypes.get_last_error()
        if code == self.ERROR_ALREADY_EXISTS:
            return False
        raise self._win_error(code)

    def close(self, handle: int) -> None:
        if not self.kernel32.CloseHandle(handle):
            raise self._win_error()

    def identity_and_attributes(self, handle: int) -> tuple[ArtifactPathIdentity, int]:
        info = self.ByHandleFileInformation()
        if not self.kernel32.GetFileInformationByHandle(
            handle, self.ctypes.byref(info)
        ):
            raise self._win_error()
        size = (int(info.nFileSizeHigh) << 32) | int(info.nFileSizeLow)
        file_index = (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow)
        file_time = (int(info.ftLastWriteTime.dwHighDateTime) << 32) | int(
            info.ftLastWriteTime.dwLowDateTime
        )
        identity = ArtifactPathIdentity(
            volume_serial=int(info.dwVolumeSerialNumber),
            file_id=file_index.to_bytes(8, "little"),
            size=size,
            mtime_ns=file_time * 100,
            link_count=int(info.nNumberOfLinks),
        )
        return identity, int(info.dwFileAttributes)

    def final_path(self, handle: int) -> Path:
        required = self.kernel32.GetFinalPathNameByHandleW(handle, None, 0, 0)
        if required == 0:
            raise self._win_error()
        buffer = self.ctypes.create_unicode_buffer(required + 1)
        written = self.kernel32.GetFinalPathNameByHandleW(
            handle, buffer, len(buffer), 0
        )
        if written == 0 or written >= len(buffer):
            raise self._win_error()
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value)

    def to_stream(self, handle: int, *, writable: bool = False) -> BinaryIO:
        import msvcrt

        flags = os.O_BINARY | (os.O_RDWR if writable else os.O_RDONLY)
        descriptor = msvcrt.open_osfhandle(handle, flags)
        return os.fdopen(descriptor, "r+b" if writable else "rb", buffering=0)

    def flush(self, handle: int) -> None:
        if not self.kernel32.FlushFileBuffers(handle):
            raise self._win_error()

    def move_create_new(self, source: Path, destination: Path) -> None:
        if not self.kernel32.MoveFileExW(
            str(source), str(destination), self.MOVEFILE_WRITE_THROUGH
        ):
            code = self.ctypes.get_last_error()
            if code in (self.ERROR_FILE_EXISTS, self.ERROR_ALREADY_EXISTS):
                raise ArtifactIntegrityError("colision con artefacto existente")
            raise self._win_error(code)

    def set_delete_disposition(self, handle: int) -> None:
        disposition = self.FileDispositionInfo(True)
        if not self.kernel32.SetFileInformationByHandle(
            handle,
            4,
            self.ctypes.byref(disposition),
            self.ctypes.sizeof(disposition),
        ):
            raise self._win_error()

    def acquire_lock(self, handle: int, *, exclusive: bool) -> object:
        overlapped = self.Overlapped()
        flags = self.LOCKFILE_FAIL_IMMEDIATELY
        if exclusive:
            flags |= self.LOCKFILE_EXCLUSIVE_LOCK
        if not self.kernel32.LockFileEx(
            handle, flags, 0, 1, 0, self.ctypes.byref(overlapped)
        ):
            raise self._win_error()
        return overlapped

    def release_lock(self, handle: int, overlapped: object) -> None:
        if not self.kernel32.UnlockFileEx(
            handle, 0, 1, 0, self.ctypes.byref(overlapped)
        ):
            raise self._win_error()

    def _current_user_sid(self) -> bytes:
        token = self.wintypes.HANDLE()
        if not self.advapi32.OpenProcessToken(
            self.kernel32.GetCurrentProcess(), 0x0008, self.ctypes.byref(token)
        ):
            raise self._win_error()
        try:
            needed = self.wintypes.DWORD()
            self.advapi32.GetTokenInformation(
                token, 1, None, 0, self.ctypes.byref(needed)
            )
            if needed.value == 0:
                raise self._win_error()
            buffer = self.ctypes.create_string_buffer(needed.value)
            if not self.advapi32.GetTokenInformation(
                token,
                1,
                buffer,
                needed,
                self.ctypes.byref(needed),
            ):
                raise self._win_error()
            token_user = self.ctypes.cast(
                buffer, self.ctypes.POINTER(self.TokenUser)
            ).contents
            sid_length = int(self.advapi32.GetLengthSid(token_user.User.Sid))
            if sid_length <= 0:
                raise self._win_error()
            return self.ctypes.string_at(token_user.User.Sid, sid_length)
        finally:
            self.close(int(token.value))

    def acl_current_user_only(self, handle: int, path: Path) -> bool:
        del path
        user_sid = self._current_user_sid()
        owner = self.wintypes.LPVOID()
        dacl = self.wintypes.LPVOID()
        descriptor = self.wintypes.LPVOID()
        result = self.advapi32.GetSecurityInfo(
            handle,
            1,
            0x00000001 | 0x00000004,
            self.ctypes.byref(owner),
            None,
            self.ctypes.byref(dacl),
            None,
            self.ctypes.byref(descriptor),
        )
        if result != 0:
            raise self._win_error(int(result))
        try:
            if not owner or not dacl:
                return False
            control = self.wintypes.WORD()
            revision = self.wintypes.DWORD()
            if not self.advapi32.GetSecurityDescriptorControl(
                descriptor,
                self.ctypes.byref(control),
                self.ctypes.byref(revision),
            ):
                raise self._win_error()
            if not int(control.value) & 0x1000:
                return False
            owner_length = int(self.advapi32.GetLengthSid(owner))
            if self.ctypes.string_at(owner, owner_length) != user_sid:
                return False
            info = self.AclSizeInformation()
            if not self.advapi32.GetAclInformation(
                dacl,
                self.ctypes.byref(info),
                self.ctypes.sizeof(info),
                2,
            ):
                raise self._win_error()
            user_allow = False
            for index in range(int(info.AceCount)):
                ace_pointer = self.wintypes.LPVOID()
                if not self.advapi32.GetAce(
                    dacl, index, self.ctypes.byref(ace_pointer)
                ):
                    raise self._win_error()
                header = self.ctypes.cast(
                    ace_pointer, self.ctypes.POINTER(self.AceHeader)
                ).contents
                if int(header.AceType) == 0:
                    sid_pointer = self.ctypes.c_void_p(
                        int(ace_pointer.value) + self.AccessAllowedAce.SidStart.offset
                    )
                    sid_length = int(self.advapi32.GetLengthSid(sid_pointer))
                    if self.ctypes.string_at(sid_pointer, sid_length) != user_sid:
                        return False
                    user_allow = True
                elif int(header.AceType) in {4, 5, 9, 11}:
                    return False
                elif int(header.AceType) not in {
                    1,
                    2,
                    3,
                    6,
                    7,
                    8,
                    10,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                }:
                    return False
            return user_allow
        finally:
            self.kernel32.LocalFree(descriptor)

    def encryption_provider(self, handle: int, path: Path) -> str | None:
        del path
        _, attributes = self.identity_and_attributes(handle)
        if attributes & self.FILE_ATTRIBUTE_ENCRYPTED:
            return "windows-efs"
        return None

    def secure_new_file(self, handle: int, path: Path) -> None:
        sid_buffer = self.ctypes.create_string_buffer(self._current_user_sid())
        sid_pointer = self.ctypes.cast(sid_buffer, self.wintypes.LPVOID)
        sid_string = self.wintypes.LPWSTR()
        if not self.advapi32.ConvertSidToStringSidW(
            sid_pointer, self.ctypes.byref(sid_string)
        ):
            raise self._win_error()
        descriptor = self.wintypes.LPVOID()
        try:
            sddl = f"D:P(A;;FA;;;{sid_string.value})"
            if not self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, self.ctypes.byref(descriptor), None
            ):
                raise self._win_error()
            present = self.wintypes.BOOL()
            defaulted = self.wintypes.BOOL()
            dacl = self.wintypes.LPVOID()
            if not self.advapi32.GetSecurityDescriptorDacl(
                descriptor,
                self.ctypes.byref(present),
                self.ctypes.byref(dacl),
                self.ctypes.byref(defaulted),
            ):
                raise self._win_error()
            if not present.value or not dacl:
                raise ArtifactIntegrityError("acl privada no pudo construirse")
            result = self.advapi32.SetSecurityInfo(
                handle,
                1,
                0x00000004 | 0x80000000,
                None,
                None,
                dacl,
                None,
            )
            if result != 0:
                raise self._win_error(int(result))
            # EFS: objects born inside an encrypted store directory inherit
            # FILE_ATTRIBUTE_ENCRYPTED; otherwise use EncryptFileW, the
            # user-mode EFS API (raw FSCTL_SET_ENCRYPTION is reserved for the
            # EFS service and always fails with ACCESS_DENIED from user
            # mode). EncryptFileW is path-based, so the attribute is
            # re-checked on the HELD handle to fail closed on a path swap.
            _, attributes = self.identity_and_attributes(handle)
            if not attributes & self.FILE_ATTRIBUTE_ENCRYPTED:
                if not self.advapi32.EncryptFileW(str(path)):
                    # EFS unavailable on some volumes/editions: promotion must
                    # fail with a typed error naming the missing property.
                    raise ArtifactIntegrityError(
                        "encryption EFS no disponible en el volumen del store"
                    ) from self._win_error()
                _, attributes = self.identity_and_attributes(handle)
                if not attributes & self.FILE_ATTRIBUTE_ENCRYPTED:
                    raise ArtifactIntegrityError(
                        "encryption EFS no demostrada sobre el handle asegurado"
                    )
        finally:
            if descriptor:
                self.kernel32.LocalFree(descriptor)
            if sid_string:
                self.kernel32.LocalFree(sid_string)


def _default_known_local_app_data() -> Path:
    import ctypes
    from ctypes import wintypes
    import uuid

    class Guid(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    value = uuid.UUID("f1b32785-6fba-4fcf-9d55-7b8e7f157091")
    guid = Guid(
        value.time_low,
        value.time_mid,
        value.time_hi_version,
        (ctypes.c_ubyte * 8)(*value.bytes[8:]),
    )
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    shell32.SHGetKnownFolderPath.argtypes = [
        ctypes.POINTER(Guid),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [wintypes.LPVOID]
    path_pointer = wintypes.LPWSTR()
    result = shell32.SHGetKnownFolderPath(
        ctypes.byref(guid), 0, None, ctypes.byref(path_pointer)
    )
    if result != 0:
        raise ctypes.WinError(result)
    try:
        return Path(path_pointer.value)
    finally:
        ole32.CoTaskMemFree(path_pointer)


class WindowsPrivateArtifactFilesystem:
    """Handle-based Windows adapter with strict ACL and encryption defaults."""

    def __init__(
        self,
        *,
        known_folder_probe: Callable[[], Path] | None = None,
        acl_probe: Callable[[int, Path], bool] | None = None,
        encryption_probe: Callable[[int, Path], str | None] | None = None,
        secure_new_file: Callable[[int, Path], None] | None = None,
    ) -> None:
        if sys.platform != "win32":
            raise ArtifactIntegrityError(
                "el adapter Win32 no existe en esta plataforma"
            )
        self._api = _Win32Api()
        self._known_folder_probe = known_folder_probe or _default_known_local_app_data
        self._acl_probe = acl_probe or self._api.acl_current_user_only
        self._encryption_probe = encryption_probe or self._api.encryption_provider
        self._secure_new_file = secure_new_file or self._api.secure_new_file

    @staticmethod
    def _same_path(left: Path, right: Path) -> bool:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
            os.path.abspath(right)
        )

    def _verify_handle(
        self,
        handle: int,
        path: Path,
        *,
        directory: bool,
        label: str,
        owned: bool = True,
    ) -> tuple[ArtifactPathIdentity, str]:
        """Verify one chain component.

        Reparse/type/identity/final-path checks run on EVERY component. The
        strict ACL (current-user-only, protected) and encryption probes run
        only on components the store owns (speechtotext, artifacts,
        subdirectories, artifact/temp/lock files) and on the promotion source
        FILE. %LOCALAPPDATA% is the OS-trusted anchor from
        SHGetKnownFolderPath and source ancestors/volume roots can never be
        current-user-only or encrypted, so probing them (owned=False) would
        make the default configuration fail closed on every real machine.
        """
        identity, attributes = self._api.identity_and_attributes(handle)
        final_path = self._api.final_path(handle)
        if not self._same_path(final_path, path):
            raise ArtifactIntegrityError(f"{label}: identidad handle-based invalida")
        if attributes & self._api.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ArtifactIntegrityError(f"{label}: reparse no permitido")
        is_directory = bool(attributes & self._api.FILE_ATTRIBUTE_DIRECTORY)
        if is_directory != directory:
            raise ArtifactIntegrityError(f"{label}: tipo de archivo invalido")
        if not directory and identity.link_count != 1:
            raise ArtifactIntegrityError(f"{label}: hardlink no permitido")
        if not owned:
            return identity, ""
        if self._acl_probe(handle, path) is not True:
            raise ArtifactIntegrityError(f"{label}: acl no demostrada")
        provider = self._encryption_probe(handle, path)
        if type(provider) is not str or not provider.strip():
            raise ArtifactIntegrityError(f"{label}: encryption no demostrada")
        return identity, provider

    def _fixed_paths(self) -> tuple[Path, Path, Path]:
        local = Path(self._known_folder_probe())
        if not local.is_absolute() or str(local).startswith(("\\\\?\\", "\\\\.\\")):
            raise ArtifactIntegrityError("Known Folder devolvio una ruta invalida")
        return local, local / "speechtotext", local / "speechtotext" / "artifacts"

    @contextmanager
    def lease_current_user_root(self) -> Iterator[ArtifactRootLease]:
        paths = self._fixed_paths()
        handles: list[int] = []
        identities: list[ArtifactPathIdentity] = []
        providers: list[str] = []
        # %LOCALAPPDATA% is the OS-trusted anchor: identity/reparse only.
        components = (
            ("known_local_app_data", False),
            ("speechtotext", True),
            ("artifacts", True),
        )
        try:
            for path, (label, owned) in zip(paths, components, strict=True):
                handle = self._api.open(
                    path,
                    access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
                    share=self._api.FILE_SHARE_READ,
                    creation=self._api.OPEN_EXISTING,
                    directory=True,
                )
                handles.append(handle)
                identity, provider = self._verify_handle(
                    handle, path, directory=True, label=label, owned=owned
                )
                identities.append(identity)
                if owned:
                    providers.append(provider)
            lease = ArtifactRootLease(
                root=paths[-1],
                path_chain=tuple(identities),
                encryption_provider="+".join(providers),
                _handles=tuple(handles),
            )
            try:
                yield lease
            finally:
                lease._state.active = False
        finally:
            for handle in reversed(handles):
                self._api.close(handle)

    def _open_secure_directory(
        self, path: Path, label: str, *, create: bool
    ) -> int:
        """Open (or provision, promotion-only) a store-owned directory.

        create=True applies CREATE_NEW semantics: a newly created directory is
        immediately secured (protected user-only DACL + EFS) on its handle and
        re-verified; on ERROR_ALREADY_EXISTS it converges to open+verify so
        concurrent creators end up on the same verified directory. Readers
        always call with create=False and never create anything.
        """
        created = create and self._api.create_directory(path)
        if created:
            self._secure_empty_directory(path)
        handle = self._api.open(
            path,
            access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
            share=self._api.FILE_SHARE_READ,
            creation=self._api.OPEN_EXISTING,
            directory=True,
        )
        try:
            try:
                self._verify_handle(handle, path, directory=True, label=label)
            except ArtifactIntegrityError:
                if not create or created:
                    raise
                # Convergence (exclusive promotion path only): a hard crash
                # between CREATE_NEW and secure_new_file leaves an unsecured
                # EMPTY directory that would wedge every later promotion.
                # Re-secure it and re-verify on the reader handle held across
                # the whole repair. Tampering shapes (reparse/type/final-path)
                # re-raise via the owned=False check, NON-EMPTY directories
                # keep failing closed, and readers (create=False) never
                # reach this branch.
                self._verify_handle(
                    handle, path, directory=True, label=label, owned=False
                )
                with os.scandir(path) as entries:
                    if next(entries, None) is not None:
                        raise
                self._secure_empty_directory(path)
                self._verify_handle(handle, path, directory=True, label=label)
            return handle
        except BaseException:
            self._api.close(handle)
            raise

    def _secure_empty_directory(self, path: Path) -> None:
        # Secure on a transient write handle and close it: a held
        # write-access directory handle registers for sharing and would
        # make the later MoveFileExW into this directory fail. Full
        # sharing here lets EncryptFileW (which reopens by path) work;
        # callers re-verify on a separate reader handle afterwards.
        secure_handle = self._api.open(
            path,
            access=(
                self._api.GENERIC_READ
                | self._api.GENERIC_WRITE
                | self._api.WRITE_DAC
                | self._api.DELETE
                | self._api.FILE_READ_ATTRIBUTES
                | self._api.READ_CONTROL
            ),
            share=(
                self._api.FILE_SHARE_READ
                | self._api.FILE_SHARE_WRITE
                | self._api.FILE_SHARE_DELETE
            ),
            creation=self._api.OPEN_EXISTING,
            directory=True,
        )
        try:
            try:
                self._secure_new_file(secure_handle, path)
            except BaseException:
                # Do not leave a half-secured directory that would wedge
                # every later promotion; it is empty, so this succeeds.
                try:
                    self._api.set_delete_disposition(secure_handle)
                except OSError:
                    pass
                raise
        finally:
            self._api.close(secure_handle)

    def _open_artifact_directories(
        self, root: ArtifactRootLease, relative_name: str, *, create: bool = False
    ) -> tuple[list[int], Path]:
        if not root._state.active:
            raise ArtifactIntegrityError("root lease no esta activo")
        parts = PurePosixPath(relative_name).parts
        handles: list[int] = []
        current = root.root
        try:
            for index, part in enumerate(parts[:-1]):
                current /= part
                handles.append(
                    self._open_secure_directory(
                        current,
                        f"artifact_component_{index}",
                        create=create,
                    )
                )
            return handles, current / parts[-1]
        except BaseException:
            for handle in reversed(handles):
                self._api.close(handle)
            raise

    @contextmanager
    def lease_file(
        self, relative_name: str, root: ArtifactRootLease
    ) -> Iterator[ArtifactFileHandle]:
        name = _safe_relative(relative_name)
        directories, path = self._open_artifact_directories(root, name)
        stream: BinaryIO | None = None
        handle: int | None = None
        try:
            handle = self._api.open(
                path,
                access=(
                    self._api.GENERIC_READ
                    | self._api.FILE_READ_ATTRIBUTES
                    | self._api.READ_CONTROL
                ),
                share=self._api.FILE_SHARE_READ,
                creation=self._api.OPEN_EXISTING,
            )
            identity, provider = self._verify_handle(
                handle, path, directory=False, label="artifact_file"
            )
            stream = self._api.to_stream(handle)
            handle = None
            yield ArtifactFileHandle(
                relative_name=name,
                stream=stream,
                identity=identity,
                encryption_provider=provider,
            )
        finally:
            if stream is not None:
                stream.close()
            elif handle is not None:
                self._api.close(handle)
            for directory_handle in reversed(directories):
                self._api.close(directory_handle)

    @staticmethod
    def _validate_source_path(source: Path) -> tuple[Path, tuple[str, ...]]:
        raw = os.fspath(source)
        if not raw or "\x00" in raw or raw.startswith(("\\\\?\\", "\\\\.\\")):
            raise ValueError("source exige una ruta privada segura")
        path = Path(raw)
        if not path.is_absolute() or not path.drive or raw.startswith("\\\\"):
            raise ValueError("source exige una ruta privada segura")
        if ":" in raw[2:]:
            raise ValueError("source exige una ruta privada segura")
        parts = path.parts[1:]
        _safe_relative("/".join(parts), allow_dot_prefix=True)
        return path, parts

    @contextmanager
    def lease_private_source(self, source: Path) -> Iterator[ArtifactSourceHandle]:
        path, parts = self._validate_source_path(Path(source))
        volume = Path(path.anchor)
        current = volume
        directory_handles: list[int] = []
        stream: BinaryIO | None = None
        file_handle: int | None = None
        try:
            volume_handle = self._api.open(
                volume,
                access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
                share=self._api.FILE_SHARE_READ,
                creation=self._api.OPEN_EXISTING,
                directory=True,
            )
            directory_handles.append(volume_handle)
            # Source ancestors (volume root included) can never be
            # current-user-only or encrypted: reparse/identity checks only.
            # The strict probes run on the source FILE below.
            self._verify_handle(
                volume_handle,
                volume,
                directory=True,
                label="source_volume",
                owned=False,
            )
            for index, part in enumerate(parts[:-1]):
                current /= part
                handle = self._api.open(
                    current,
                    access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
                    share=self._api.FILE_SHARE_READ,
                    creation=self._api.OPEN_EXISTING,
                    directory=True,
                )
                directory_handles.append(handle)
                self._verify_handle(
                    handle,
                    current,
                    directory=True,
                    label=f"source_component_{index}",
                    owned=False,
                )
            file_handle = self._api.open(
                path,
                access=(
                    self._api.GENERIC_READ
                    | self._api.FILE_READ_ATTRIBUTES
                    | self._api.READ_CONTROL
                ),
                share=self._api.FILE_SHARE_READ,
                creation=self._api.OPEN_EXISTING,
            )
            identity, provider = self._verify_handle(
                file_handle, path, directory=False, label="source_file"
            )
            stream = self._api.to_stream(file_handle)
            file_handle = None
            yield ArtifactSourceHandle(
                stream=stream,
                identity=identity,
                encryption_provider=provider,
            )
        finally:
            if stream is not None:
                stream.close()
            elif file_handle is not None:
                self._api.close(file_handle)
            for handle in reversed(directory_handles):
                self._api.close(handle)

    def _provision_store(self) -> None:
        """Create missing store directories (offline promotion only).

        %LOCALAPPDATA% must already exist -- it is the OS anchor and is never
        created here. speechtotext and artifacts are created with CREATE_NEW
        semantics, secured (protected user-only DACL + EFS) on the fresh
        handle and re-verified; a preexisting component is accepted only if
        it passes the probes. Readers never provision.
        """
        local, speech, artifacts_root = self._fixed_paths()
        handle = self._api.open(
            local,
            access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
            share=self._api.FILE_SHARE_READ,
            creation=self._api.OPEN_EXISTING,
            directory=True,
        )
        try:
            self._verify_handle(
                handle,
                local,
                directory=True,
                label="known_local_app_data",
                owned=False,
            )
        finally:
            self._api.close(handle)
        for path, label in ((speech, "speechtotext"), (artifacts_root, "artifacts")):
            self._api.close(self._open_secure_directory(path, label, create=True))

    @contextmanager
    def _lock(self, *, exclusive: bool) -> Iterator[None]:
        if exclusive:
            self._provision_store()
        with self.lease_current_user_root() as root:
            # Brief 1A: the shared/exclusive OS lock lives under the fixed
            # parent %LOCALAPPDATA%\speechtotext, not inside the artifacts
            # root, so it is never co-located with promoted artifacts.
            path = root.root.parent / ".runtime.lock"
            handle: int | None = None
            overlapped: object | None = None
            created = False
            validated = False
            try:
                access = (
                    self._api.GENERIC_READ
                    | self._api.GENERIC_WRITE
                    | self._api.DELETE
                    | self._api.READ_CONTROL
                    | self._api.WRITE_DAC
                )
                share = (
                    self._api.FILE_SHARE_READ
                    | self._api.FILE_SHARE_WRITE
                    | self._api.FILE_SHARE_DELETE
                )
                try:
                    handle = self._api.open(
                        path,
                        access=access,
                        share=share,
                        creation=self._api.CREATE_NEW,
                    )
                    created = True
                except OSError as exc:
                    if getattr(exc, "winerror", None) not in {
                        self._api.ERROR_FILE_EXISTS,
                        self._api.ERROR_ALREADY_EXISTS,
                    }:
                        raise
                    handle = self._api.open(
                        path,
                        access=access,
                        share=share,
                        creation=self._api.OPEN_EXISTING,
                    )
                if created:
                    self._secure_new_file(handle, path)
                try:
                    self._verify_handle(
                        handle, path, directory=False, label="runtime_lock"
                    )
                except ArtifactIntegrityError:
                    if created or not exclusive:
                        raise
                    # Convergence (exclusive promotion path only): a hard
                    # crash between CREATE_NEW and secure_new_file leaves an
                    # unsecured lock that would wedge every later session.
                    # The held handle already carries WRITE_DAC: re-secure it
                    # and re-verify. Tampering shapes (reparse/type/hardlink)
                    # re-raise via the owned=False check; readers
                    # (runtime_session) never self-heal.
                    self._verify_handle(
                        handle,
                        path,
                        directory=False,
                        label="runtime_lock",
                        owned=False,
                    )
                    self._secure_new_file(handle, path)
                    self._verify_handle(
                        handle, path, directory=False, label="runtime_lock"
                    )
                validated = True
                try:
                    overlapped = self._api.acquire_lock(handle, exclusive=exclusive)
                except OSError as exc:
                    if getattr(exc, "winerror", None) in {
                        self._api.ERROR_LOCK_VIOLATION,
                        self._api.ERROR_SHARING_VIOLATION,
                    }:
                        raise ArtifactIntegrityError(
                            "servicio activo impide promocion"
                        ) from None
                    raise
                if exclusive:
                    self._cleanup_stale_temps(root)
                yield None
            finally:
                if handle is not None and overlapped is not None:
                    self._api.release_lock(handle, overlapped)
                if handle is not None and created and not validated:
                    try:
                        self._api.set_delete_disposition(handle)
                    except OSError:
                        pass
                if handle is not None:
                    self._api.close(handle)

    def runtime_session(self) -> AbstractContextManager[None]:
        return self._lock(exclusive=False)

    def offline_promotion(self) -> AbstractContextManager[None]:
        return self._lock(exclusive=True)

    def _remove_matching(
        self, path: Path, expected_identity: ArtifactPathIdentity
    ) -> None:
        handle: int | None = None
        try:
            handle = self._api.open(
                path,
                access=(
                    self._api.DELETE
                    | self._api.FILE_READ_ATTRIBUTES
                    | self._api.READ_CONTROL
                ),
                share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_DELETE,
                creation=self._api.OPEN_EXISTING,
            )
            identity, _ = self._api.identity_and_attributes(handle)
            if identity != expected_identity:
                raise ArtifactIntegrityError(
                    "identidad cambio; no se elimina un path ajeno"
                )
            self._api.set_delete_disposition(handle)
        finally:
            if handle is not None:
                self._api.close(handle)

    def _cleanup_stale_temps(self, root: ArtifactRootLease) -> None:
        """Delete stale promotion temps under the exclusive lock.

        A crash between CREATE_NEW and secure_new_file leaves a temp with an
        inherited DACL and no EFS attribute, so the ACL/encryption probes must
        NOT run here: a temp's own verification failure would wedge every
        future promotion. Only reparse/regular-file/link-count are checked
        before deleting through the DELETE-access handle.
        """
        for path in root.root.iterdir():
            if not _PRIVATE_TEMP_RE.fullmatch(path.name):
                continue
            handle: int | None = None
            try:
                handle = self._api.open(
                    path,
                    access=(
                        self._api.DELETE
                        | self._api.FILE_READ_ATTRIBUTES
                        | self._api.READ_CONTROL
                    ),
                    share=(self._api.FILE_SHARE_READ | self._api.FILE_SHARE_DELETE),
                    creation=self._api.OPEN_EXISTING,
                )
                identity, attributes = self._api.identity_and_attributes(handle)
                if attributes & self._api.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise ArtifactIntegrityError(
                        "stale_promotion_temp: reparse no permitido"
                    )
                if attributes & self._api.FILE_ATTRIBUTE_DIRECTORY:
                    raise ArtifactIntegrityError(
                        "stale_promotion_temp: tipo de archivo invalido"
                    )
                if identity.link_count != 1:
                    raise ArtifactIntegrityError(
                        "stale_promotion_temp: hardlink no permitido"
                    )
                self._api.set_delete_disposition(handle)
            finally:
                if handle is not None:
                    self._api.close(handle)

    def install_create_new(
        self,
        relative_name: str,
        root: ArtifactRootLease,
        source: ArtifactSourceHandle,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> ArtifactPathIdentity:
        name = _safe_relative(relative_name)
        # Promotion is the exclusive path: missing destination subdirectories
        # are provisioned (CREATE_NEW + secure + verify) here.
        directories, destination = self._open_artifact_directories(
            root, name, create=True
        )
        import secrets

        temp = root.root / f".artifact-{secrets.token_hex(16)}.tmp"
        stream: BinaryIO | None = None
        native_handle: int | None = None
        temp_identity: ArtifactPathIdentity | None = None
        published = False
        try:
            native_handle = self._api.open(
                temp,
                access=(
                    self._api.GENERIC_READ
                    | self._api.GENERIC_WRITE
                    | self._api.DELETE
                    | self._api.FILE_READ_ATTRIBUTES
                    | self._api.READ_CONTROL
                    | self._api.WRITE_DAC
                ),
                share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_DELETE,
                creation=self._api.CREATE_NEW,
            )
            temp_identity, _ = self._api.identity_and_attributes(native_handle)
            self._secure_new_file(native_handle, temp)
            temp_identity, _ = self._verify_handle(
                native_handle, temp, directory=False, label="promotion_temp"
            )
            stream = self._api.to_stream(native_handle, writable=True)
            native_handle = None
            copied, source_digest = _copy_bounded_with_sha256(
                source.stream, stream, max_bytes
            )
            if copied != source.identity.size:
                raise ArtifactIntegrityError("identidad de source cambio durante copia")
            if not hmac.compare_digest(source_digest, expected_sha256):
                raise ArtifactIntegrityError("sha256 de source no coincide")
            stream.flush()
            os.fsync(stream.fileno())
            import msvcrt

            live_handle = int(msvcrt.get_osfhandle(stream.fileno()))
            self._api.flush(live_handle)
            temp_identity, _ = self._api.identity_and_attributes(live_handle)
            # MOVEFILE_WRITE_THROUGH makes the rename itself durable; a
            # FlushFileBuffers on the attribute-only directory handles always
            # failed (no write access), so no directory flush is attempted.
            self._api.move_create_new(temp, destination)
            published = True
            stream.close()
            stream = None

            with self.lease_file(name, root) as destination_handle:
                if destination_handle.identity != temp_identity:
                    raise ArtifactIntegrityError(
                        "identidad de destino no coincide tras reopen"
                    )
                payload_after = _read_bounded(destination_handle.stream, max_bytes)
                if not hmac.compare_digest(
                    hashlib.sha256(payload_after).hexdigest(), expected_sha256
                ):
                    raise ArtifactIntegrityError(
                        "sha256 de destino no coincide tras reopen"
                    )
                if len(payload_after) != temp_identity.size:
                    raise ArtifactIntegrityError(
                        "identidad de destino no coincide tras reopen"
                    )
                identity = destination_handle.identity
            return identity
        except BaseException:
            cleanup_path = destination if published else temp
            cleanup_done = False
            if stream is not None:
                if not published:
                    import msvcrt

                    try:
                        self._api.set_delete_disposition(
                            int(msvcrt.get_osfhandle(stream.fileno()))
                        )
                        cleanup_done = True
                    except OSError:
                        pass
                stream.close()
                stream = None
            elif native_handle is not None:
                if not published:
                    try:
                        self._api.set_delete_disposition(native_handle)
                        cleanup_done = True
                    except OSError:
                        pass
                self._api.close(native_handle)
                native_handle = None
            if temp_identity is not None and not cleanup_done:
                try:
                    self._remove_matching(cleanup_path, temp_identity)
                except OSError:
                    pass
            raise
        finally:
            if stream is not None:
                stream.close()
            elif native_handle is not None:
                self._api.close(native_handle)
            for handle in reversed(directories):
                self._api.close(handle)


def default_private_artifact_filesystem() -> PrivateArtifactFilesystem:
    if sys.platform != "win32":
        raise ArtifactIntegrityError(
            "el store privado requiere un adapter verificable de Windows"
        )
    return WindowsPrivateArtifactFilesystem()
