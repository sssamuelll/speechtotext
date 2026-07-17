from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import BinaryIO, Protocol


_WIN_FORBIDDEN = frozenset('<>:"\\|?*')
_WIN_RESERVED = frozenset({"con", "prn", "aux", "nul"})
_WIN_DEVICE_SUFFIXES = frozenset((*"123456789", "\u00b9", "\u00b2", "\u00b3"))


class ModelFilesystemError(RuntimeError):
    """The OS could not demonstrate an immutable model read."""


@dataclass(frozen=True)
class ModelPathIdentity:
    path: Path
    volume_serial: int
    file_id: bytes


@dataclass
class _RootState:
    active: bool = True


@dataclass(frozen=True)
class ModelRootLease:
    root: Path
    volume_serial: int
    file_id: bytes
    read_only: bool
    path_chain: tuple[ModelPathIdentity, ...]
    _state: _RootState = field(
        default_factory=_RootState, repr=False, compare=False
    )
    _handles: tuple[object, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class ModelFileLease:
    relative_path: str
    stream: BinaryIO
    volume_serial: int
    file_id: bytes
    size: int
    mtime_ns: int


class ModelFilesystem(Protocol):
    def lease_read_only_root(
        self, root: Path
    ) -> AbstractContextManager[ModelRootLease]: ...

    def relative_path(self, path: Path, root: ModelRootLease) -> str: ...

    def inventory(self, root: ModelRootLease) -> tuple[str, ...]: ...

    def lease_file(
        self, relative_path: str, root: ModelRootLease
    ) -> AbstractContextManager[ModelFileLease]: ...


def _safe_relative(value: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("el modelo exige una ruta relativa segura")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("el modelo exige una ruta relativa segura")
    for part in path.parts:
        stem = part.split(".", 1)[0].casefold()
        device = stem in _WIN_RESERVED or (
            len(stem) == 4
            and stem[:3] in {"com", "lpt"}
            and stem[3] in _WIN_DEVICE_SUFFIXES
        )
        if (
            part.endswith((".", " "))
            or any(ord(char) < 32 or char in _WIN_FORBIDDEN for char in part)
            or device
            or len(part.encode("utf-16-le")) // 2 > 255
        ):
            raise ValueError("el modelo exige una ruta relativa segura")
    if len(value.encode("utf-16-le")) // 2 > 32767:
        raise ValueError("el modelo exige una ruta relativa segura")
    return value


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _contained(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)
    except ValueError:
        return False


class FakeModelFilesystem:
    """Stateful fake implementing the same leases as the OS adapter."""

    def __init__(self, *, root_read_only: bool) -> None:
        self.root_read_only = root_read_only
        self.leased_model_paths: set[str] = set()
        self._leased_paths: Counter[str] = Counter()
        self._active_roots: Counter[Path] = Counter()
        self._reparse: set[Path] = set()
        self._hardlinks: set[Path] = set()
        self._inventory_aliases: set[str] = set()
        self._identity_after_lease: set[Path] = set()

    @staticmethod
    def _identity(path: Path) -> tuple[int, bytes, int, int]:
        info = path.stat(follow_symlinks=False)
        file_id = int(info.st_ino).to_bytes(16, "little", signed=False)
        return int(info.st_dev), file_id, int(info.st_size), int(info.st_mtime_ns)

    def _require_active(self, root: ModelRootLease) -> None:
        if not root._state.active or not self._active_roots[root.root]:
            raise ModelFilesystemError("root lease no esta activo")

    @contextmanager
    def lease_read_only_root(self, root: Path) -> Iterator[ModelRootLease]:
        requested = Path(root).absolute()
        if not self.root_read_only:
            raise ModelFilesystemError("root de modelo no es read-only")
        if not requested.is_dir():
            raise ModelFilesystemError("root de modelo no existe")
        chain_paths = tuple(reversed((requested, *requested.parents)))
        identities = tuple(
            ModelPathIdentity(
                path=component,
                volume_serial=self._identity(component)[0],
                file_id=self._identity(component)[1],
            )
            for component in chain_paths
        )
        root_identity = identities[-1]
        lease = ModelRootLease(
            root=requested,
            volume_serial=root_identity.volume_serial,
            file_id=root_identity.file_id,
            read_only=True,
            path_chain=identities,
        )
        self._active_roots[requested] += 1
        try:
            yield lease
        finally:
            lease._state.active = False
            self._active_roots[requested] -= 1

    def relative_path(self, path: Path, root: ModelRootLease) -> str:
        self._require_active(root)
        candidate = Path(path).absolute()
        if not _contained(candidate, root.root) or _same_path(candidate, root.root):
            raise ModelFilesystemError("ruta fuera del root leased")
        relative = candidate.relative_to(root.root).as_posix()
        return _safe_relative(relative)

    def _check_tree(self, root: ModelRootLease) -> None:
        self._require_active(root)
        for component in (root.root, *root.root.parents):
            if component in self._reparse or component.is_symlink():
                raise ModelFilesystemError("reparse no permitido en root de modelo")
        current = self._identity(root.root)
        if (current[0], current[1]) != (root.volume_serial, root.file_id):
            raise ModelFilesystemError("identidad del root de modelo cambio")

    def inventory(self, root: ModelRootLease) -> tuple[str, ...]:
        self._check_tree(root)
        names: list[str] = []
        for candidate in root.root.rglob("*"):
            if candidate in self._reparse or candidate.is_symlink():
                raise ModelFilesystemError("reparse no permitido en inventario")
            if candidate.is_file():
                names.append(candidate.relative_to(root.root).as_posix())
            elif candidate.is_dir() and not any(
                item.is_file() for item in candidate.rglob("*")
            ):
                raise ModelFilesystemError(
                    "inventario de modelo con directorio extra vacio"
                )
        names.extend(self._inventory_aliases)
        folded = [name.casefold() for name in names]
        if len(folded) != len(set(folded)):
            raise ModelFilesystemError("inventario tiene colision por case-fold")
        if any(
            path in self._identity_after_lease
            and self._leased_paths[path.relative_to(root.root).as_posix()]
            for path in self._identity_after_lease
            if _contained(path, root.root)
        ):
            raise ModelFilesystemError("identidad del archivo cambio durante lease")
        return tuple(sorted(names, key=str.casefold))

    @contextmanager
    def lease_file(
        self, relative_path: str, root: ModelRootLease
    ) -> Iterator[ModelFileLease]:
        self._check_tree(root)
        name = _safe_relative(relative_path)
        path = root.root.joinpath(*PurePosixPath(name).parts)
        if not _contained(path, root.root):
            raise ModelFilesystemError("ruta fuera del root leased")
        for component in (*path.parents, path):
            if component == root.root.parent:
                break
            if component in self._reparse or component.is_symlink():
                raise ModelFilesystemError("reparse no permitido")
        if not path.is_file():
            raise ModelFilesystemError("archivo regular de modelo no encontrado")
        info = path.stat(follow_symlinks=False)
        if path in self._hardlinks or info.st_nlink != 1:
            raise ModelFilesystemError("hardlink no permitido")
        volume, file_id, size, mtime_ns = self._identity(path)
        stream = path.open("rb")
        self._leased_paths[name] += 1
        self.leased_model_paths.add(name)
        try:
            yield ModelFileLease(name, stream, volume, file_id, size, mtime_ns)
        finally:
            stream.close()
            self._leased_paths[name] -= 1
            if not self._leased_paths[name]:
                self.leased_model_paths.discard(name)

    def _blocked_by_root(self, path: Path) -> bool:
        candidate = Path(path).absolute()
        return any(
            count
            and (
                _contained(candidate, root)
                or _same_path(candidate, root)
                or root in candidate.parents
            )
            for root, count in self._active_roots.items()
        )

    def replace(self, path: Path, payload: bytes) -> None:
        if self._blocked_by_root(path):
            raise PermissionError("sharing violation")
        Path(path).write_bytes(payload)

    def create(self, path: Path, payload: bytes) -> None:
        if self._blocked_by_root(path):
            raise PermissionError("sharing violation")
        Path(path).write_bytes(payload)

    def rename_root(self, root: Path) -> None:
        if self._active_roots[Path(root).absolute()]:
            raise PermissionError("sharing violation")

    def replace_ancestor(self, ancestor: Path) -> None:
        candidate = Path(ancestor).absolute()
        if any(
            count and candidate in root.parents
            for root, count in self._active_roots.items()
        ):
            raise PermissionError("sharing violation")

    def mark_reparse(self, path: Path) -> None:
        self._reparse.add(Path(path).absolute())

    def mark_hardlink(self, path: Path) -> None:
        self._hardlinks.add(Path(path).absolute())

    def add_inventory_alias(self, relative_path: str) -> None:
        self._inventory_aliases.add(_safe_relative(relative_path))

    def change_identity_after_lease(self, path: Path) -> None:
        self._identity_after_lease.add(Path(path).absolute())

    def clear_faults(self) -> None:
        self._reparse.clear()
        self._hardlinks.clear()
        self._inventory_aliases.clear()
        self._identity_after_lease.clear()


@dataclass(frozen=True)
class _WindowsIdentity:
    volume_serial: int
    file_id: bytes
    size: int
    mtime_ns: int
    link_count: int


class _WindowsApi:
    GENERIC_READ = 0x80000000
    READ_CONTROL = 0x00020000
    FILE_READ_ATTRIBUTES = 0x00000080
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
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

        class SidAndAttributes(ctypes.Structure):
            _fields_ = [
                ("Sid", wintypes.LPVOID),
                ("Attributes", wintypes.DWORD),
            ]

        class TokenUser(ctypes.Structure):
            _fields_ = [("User", SidAndAttributes)]

        self.ByHandleFileInformation = ByHandleFileInformation
        self.AclSizeInformation = AclSizeInformation
        self.AceHeader = AceHeader
        self.TokenUser = TokenUser
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
        self.advapi32.GetSecurityDescriptorControl.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.WORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
        self.advapi32.ConvertSidToStringSidW.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.LPWSTR),
        ]
        self.advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
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
        self.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
        self.kernel32.LocalFree.restype = wintypes.HLOCAL

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
        directory: bool,
    ) -> int:
        flags = self.FILE_ATTRIBUTE_NORMAL | self.FILE_FLAG_OPEN_REPARSE_POINT
        if directory:
            flags |= self.FILE_FLAG_BACKUP_SEMANTICS
        handle = self.kernel32.CreateFileW(
            str(path), access, share, None, self.OPEN_EXISTING, flags, None
        )
        invalid = self.ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise self._error()
        return int(handle)

    def close(self, handle: int) -> None:
        if not self.kernel32.CloseHandle(handle):
            raise self._error()

    def identity(self, handle: int) -> tuple[_WindowsIdentity, int]:
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
        identity = _WindowsIdentity(
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

    # Owners that cannot be re-ACLed by an unprivileged local user. Any other
    # owner keeps implicit WRITE_DAC over the object, so a read-only DACL it
    # owns proves nothing durable. Policy: accept SYSTEM / Administrators /
    # TrustedInstaller as owners outright; accept the current user as owner
    # only when the DACL is SE_DACL_PROTECTED, so at least no inheritable ACE
    # can silently widen the tree. Anything else fails closed.
    _TRUSTED_OWNER_SIDS = frozenset(
        {
            "S-1-5-18",
            "S-1-5-32-544",
            "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464",
        }
    )

    def _string_sid(self, sid_pointer) -> str:
        rendered = self.wintypes.LPWSTR()
        if not self.advapi32.ConvertSidToStringSidW(
            sid_pointer, self.ctypes.byref(rendered)
        ):
            raise self._error()
        try:
            return str(rendered.value)
        finally:
            self.kernel32.LocalFree(rendered)

    def _current_user_string_sid(self) -> str:
        token = self.wintypes.HANDLE()
        if not self.advapi32.OpenProcessToken(
            self.kernel32.GetCurrentProcess(), 0x0008, self.ctypes.byref(token)
        ):
            raise self._error()
        try:
            needed = self.wintypes.DWORD()
            self.advapi32.GetTokenInformation(
                token, 1, None, 0, self.ctypes.byref(needed)
            )
            if needed.value == 0:
                raise self._error()
            buffer = self.ctypes.create_string_buffer(needed.value)
            if not self.advapi32.GetTokenInformation(
                token, 1, buffer, needed, self.ctypes.byref(needed)
            ):
                raise self._error()
            token_user = self.ctypes.cast(
                buffer, self.ctypes.POINTER(self.TokenUser)
            ).contents
            return self._string_sid(token_user.User.Sid)
        finally:
            self.close(int(token.value))

    def dacl_is_read_only(self, handle: int, path: Path) -> bool:
        del path
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
        if result:
            raise self._error(int(result))
        try:
            if not dacl or not owner:
                return False
            control = self.wintypes.WORD()
            revision = self.wintypes.DWORD()
            if not self.advapi32.GetSecurityDescriptorControl(
                descriptor,
                self.ctypes.byref(control),
                self.ctypes.byref(revision),
            ):
                raise self._error()
            owner_sid = self._string_sid(owner)
            if owner_sid not in self._TRUSTED_OWNER_SIDS:
                if owner_sid != self._current_user_string_sid():
                    return False
                if not int(control.value) & 0x1000:  # SE_DACL_PROTECTED
                    return False
            info = self.AclSizeInformation()
            if not self.advapi32.GetAclInformation(
                dacl, self.ctypes.byref(info), self.ctypes.sizeof(info), 2
            ):
                raise self._error()
            # Forbidden rights in any allow ACE: GENERIC_ALL, GENERIC_WRITE,
            # DELETE|WRITE_DAC|WRITE_OWNER, and the file/dir write class
            # (FILE_WRITE_DATA, FILE_APPEND_DATA, FILE_WRITE_EA,
            # FILE_DELETE_CHILD, FILE_WRITE_ATTRIBUTES). FILE_EXECUTE /
            # FILE_TRAVERSE (0x20) is a read-class right: standard
            # "Read & execute" DACLs must pass.
            forbidden = (
                0x10000000
                | 0x40000000
                | 0x000D0000
                | 0x00000156
            )
            saw_allow = False
            for index in range(int(info.AceCount)):
                pointer = self.wintypes.LPVOID()
                if not self.advapi32.GetAce(dacl, index, self.ctypes.byref(pointer)):
                    raise self._error()
                header = self.ctypes.cast(
                    pointer, self.ctypes.POINTER(self.AceHeader)
                ).contents
                if int(header.AceType) in {0, 5, 9, 11}:
                    saw_allow = True
                    mask = self.ctypes.c_uint32.from_address(
                        int(pointer.value) + 4
                    ).value
                    if mask & forbidden:
                        return False
            return saw_allow
        finally:
            if descriptor:
                self.kernel32.LocalFree(descriptor)


class WindowsModelFilesystem:
    """Windows adapter pinning the model tree with non-delete-sharing handles."""

    def __init__(
        self,
        *,
        read_only_acl_probe: Callable[[int, Path], bool] | None = None,
    ) -> None:
        if sys.platform != "win32":
            raise ModelFilesystemError("el adapter Windows no existe aqui")
        self._api = _WindowsApi()
        self._read_only_acl_probe = (
            read_only_acl_probe or self._api.dacl_is_read_only
        )

    @staticmethod
    def _absolute_local(path: Path) -> Path:
        raw = os.fspath(path)
        if (
            not raw
            or "\x00" in raw
            or raw.startswith(("\\\\?\\", "\\\\.\\", "\\\\"))
        ):
            raise ValueError("root de modelo exige una ruta local segura")
        candidate = Path(raw).absolute()
        rendered = os.fspath(candidate)
        if not candidate.is_absolute() or not candidate.drive or ":" in rendered[2:]:
            raise ValueError("root de modelo exige una ruta local segura")
        return candidate

    def _verify_handle(
        self, handle: int, path: Path, *, directory: bool, label: str
    ) -> _WindowsIdentity:
        identity, attributes = self._api.identity(handle)
        if not _same_path(self._api.final_path(handle), path):
            raise ModelFilesystemError(f"{label}: identidad handle-based invalida")
        if attributes & self._api.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ModelFilesystemError(f"{label}: reparse no permitido")
        is_directory = bool(attributes & self._api.FILE_ATTRIBUTE_DIRECTORY)
        if is_directory != directory:
            raise ModelFilesystemError(f"{label}: tipo no regular")
        if not directory and identity.link_count != 1:
            raise ModelFilesystemError(f"{label}: hardlink no permitido")
        return identity

    @staticmethod
    def _chain(path: Path) -> tuple[Path, ...]:
        volume = Path(path.anchor)
        current = volume
        result = [volume]
        for part in path.parts[1:]:
            current /= part
            result.append(current)
        return tuple(result)

    @contextmanager
    def lease_read_only_root(self, root: Path) -> Iterator[ModelRootLease]:
        requested = self._absolute_local(Path(root))
        handles: list[int] = []
        identities: list[ModelPathIdentity] = []
        paths = self._chain(requested)
        try:
            for index, path in enumerate(paths):
                final = index == len(paths) - 1
                share = self._api.FILE_SHARE_READ
                if not final:
                    share |= self._api.FILE_SHARE_WRITE
                handle = self._api.open(
                    path,
                    access=self._api.FILE_READ_ATTRIBUTES
                    | (self._api.READ_CONTROL if final else 0),
                    share=share,
                    directory=True,
                )
                handles.append(handle)
                identity = self._verify_handle(
                    handle, path, directory=True, label=f"root_component_{index}"
                )
                identities.append(
                    ModelPathIdentity(path, identity.volume_serial, identity.file_id)
                )
            root_handle = handles[-1]
            if not self._read_only_acl_probe(root_handle, requested):
                raise ModelFilesystemError("DACL read-only del root no demostrada")
            identity = identities[-1]
            lease = ModelRootLease(
                root=requested,
                volume_serial=identity.volume_serial,
                file_id=identity.file_id,
                read_only=True,
                path_chain=tuple(identities),
                _handles=tuple(handles),
            )
            try:
                yield lease
            finally:
                lease._state.active = False
        finally:
            for handle in reversed(handles):
                self._api.close(handle)

    @staticmethod
    def _require_active(root: ModelRootLease) -> None:
        if not root._state.active or not root._handles:
            raise ModelFilesystemError("root lease no esta activo")

    def relative_path(self, path: Path, root: ModelRootLease) -> str:
        self._require_active(root)
        candidate = self._absolute_local(Path(path))
        if not _contained(candidate, root.root) or _same_path(candidate, root.root):
            raise ModelFilesystemError("ruta fuera del root leased")
        return _safe_relative(candidate.relative_to(root.root).as_posix())

    def _revalidate_root(self, root: ModelRootLease) -> None:
        self._require_active(root)
        identity = self._verify_handle(
            int(root._handles[-1]), root.root, directory=True, label="model_root"
        )
        if (identity.volume_serial, identity.file_id) != (
            root.volume_serial,
            root.file_id,
        ):
            raise ModelFilesystemError("identidad del root cambio")

    def _probe_directory(self, handle: int, path: Path) -> None:
        if not self._read_only_acl_probe(handle, path):
            raise ModelFilesystemError(
                "DACL read-only de directorio del modelo no demostrada"
            )

    def _scan_directory(
        self,
        directory: Path,
        root: Path,
        names: list[str],
        handles: list[int],
    ) -> int:
        try:
            entries = tuple(os.scandir(directory))
        except OSError as error:
            raise ModelFilesystemError("inventario del modelo no pudo leerse") from error
        files = 0
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            attributes = getattr(info, "st_file_attributes", 0)
            if entry.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ModelFilesystemError("reparse no permitido en inventario")
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                # Open every subdirectory by handle (READ_CONTROL) so the
                # object we probe/verify is the object we then enumerate; the
                # handle stays open (no FILE_SHARE_DELETE) for the whole scan,
                # so the directory cannot be swapped mid-descent.
                handle = self._api.open(
                    path,
                    access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
                    share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_WRITE,
                    directory=True,
                )
                handles.append(handle)
                self._verify_handle(
                    handle, path, directory=True, label="inventory_directory"
                )
                self._probe_directory(handle, path)
                child_files = self._scan_directory(path, root, names, handles)
                if child_files == 0:
                    raise ModelFilesystemError(
                        "inventario de modelo con directorio extra vacio"
                    )
                files += child_files
            elif entry.is_file(follow_symlinks=False):
                names.append(path.relative_to(root).as_posix())
                files += 1
            else:
                raise ModelFilesystemError("inventario contiene tipo no regular")
        return files

    def inventory(self, root: ModelRootLease) -> tuple[str, ...]:
        self._revalidate_root(root)
        names: list[str] = []
        handles: list[int] = []
        try:
            self._scan_directory(root.root, root.root, names, handles)
        finally:
            for handle in reversed(handles):
                self._api.close(handle)
        folded = [name.casefold() for name in names]
        if len(folded) != len(set(folded)):
            raise ModelFilesystemError("inventario tiene colision por case-fold")
        return tuple(sorted((_safe_relative(name) for name in names), key=str.casefold))

    @contextmanager
    def lease_file(
        self, relative_path: str, root: ModelRootLease
    ) -> Iterator[ModelFileLease]:
        self._revalidate_root(root)
        name = _safe_relative(relative_path)
        parts = PurePosixPath(name).parts
        current = root.root
        directories: list[int] = []
        handle: int | None = None
        stream: BinaryIO | None = None
        try:
            for index, part in enumerate(parts[:-1]):
                current /= part
                directory_handle = self._api.open(
                    current,
                    access=self._api.FILE_READ_ATTRIBUTES | self._api.READ_CONTROL,
                    share=self._api.FILE_SHARE_READ | self._api.FILE_SHARE_WRITE,
                    directory=True,
                )
                directories.append(directory_handle)
                self._verify_handle(
                    directory_handle,
                    current,
                    directory=True,
                    label=f"model_directory_{index}",
                )
                # The read-only proof must hold for every directory the leased
                # file lives under, not just the root; these handles stay open
                # for the life of the file lease.
                self._probe_directory(directory_handle, current)
            path = current / parts[-1]
            handle = self._api.open(
                path,
                access=self._api.GENERIC_READ | self._api.FILE_READ_ATTRIBUTES,
                share=self._api.FILE_SHARE_READ,
                directory=False,
            )
            identity = self._verify_handle(
                handle, path, directory=False, label="model_file"
            )
            stream = self._api.to_stream(handle)
            handle = None
            yield ModelFileLease(
                name,
                stream,
                identity.volume_serial,
                identity.file_id,
                identity.size,
                identity.mtime_ns,
            )
        except (ValueError, ModelFilesystemError):
            raise
        except OSError as error:
            raise ModelFilesystemError(f"no se pudo leasear {name}") from error
        finally:
            if stream is not None:
                stream.close()
            elif handle is not None:
                self._api.close(handle)
            for directory_handle in reversed(directories):
                self._api.close(directory_handle)


def default_model_filesystem() -> ModelFilesystem:
    if sys.platform != "win32":
        raise ModelFilesystemError(
            "se requiere un adapter verificable; solo Windows esta disponible"
        )
    return WindowsModelFilesystem()
