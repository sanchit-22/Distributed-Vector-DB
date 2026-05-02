"""Binary write-ahead log for VecScaleDB Phase 2."""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np


OP_INSERT = 0
OP_DELETE = 1

_HEADER = struct.Struct("<QB I")  # lsn, op_type, num_vectors
_CHECKSUM = struct.Struct("<I")


@dataclass(frozen=True)
class WALRecord:
    lsn: int
    op_type: int
    ids: np.ndarray
    vectors: np.ndarray


class WAL:
    """Append-only binary WAL with checksum-based recovery."""

    def __init__(self, path: str, dim: int) -> None:
        self.path = path
        self.dim = dim
        self.file_path = self._resolve_file_path(path)
        self.meta_path = f"{self.file_path}.meta"
        self._file = None
        self._last_lsn = 0

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        meta_lsn = self._read_meta_lsn()
        recovered = self.recover()
        recovered_lsn = recovered[-1].lsn if recovered else 0
        self._last_lsn = max(meta_lsn, recovered_lsn)
        self._file = open(self.file_path, "ab")
        self._write_meta_lsn()

    def append(self, op_type: int, ids: np.ndarray, vectors: np.ndarray) -> int:
        if self._file is None:
            raise RuntimeError("WAL is not open.")
        ids, vectors = self._prepare_record_arrays(ids, vectors)
        self._last_lsn += 1
        payload = self._serialize_record(self._last_lsn, op_type, ids, vectors)
        self._file.write(payload)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._write_meta_lsn()
        return self._last_lsn

    def read_from(self, lsn: int) -> Iterator[WALRecord]:
        for record in self.recover():
            if record.lsn >= lsn:
                yield record

    def recover(self) -> list[WALRecord]:
        if not os.path.isfile(self.file_path):
            return []

        records: list[WALRecord] = []
        with open(self.file_path, "rb") as f:
            while True:
                header = f.read(_HEADER.size)
                if not header:
                    break
                if len(header) != _HEADER.size:
                    break

                lsn, op_type, count = _HEADER.unpack(header)
                ids_bytes_len = count * np.dtype(np.uint64).itemsize
                vectors_bytes_len = count * self.dim * np.dtype(np.float32).itemsize
                body = f.read(ids_bytes_len + vectors_bytes_len)
                checksum_bytes = f.read(_CHECKSUM.size)

                if len(body) != ids_bytes_len + vectors_bytes_len:
                    break
                if len(checksum_bytes) != _CHECKSUM.size:
                    break

                expected = _CHECKSUM.unpack(checksum_bytes)[0]
                actual = zlib.crc32(header + body) & 0xFFFFFFFF
                if expected != actual:
                    break

                ids_buf = body[:ids_bytes_len]
                vectors_buf = body[ids_bytes_len:]
                ids = np.frombuffer(ids_buf, dtype=np.uint64).astype(np.int64)
                vectors = np.frombuffer(vectors_buf, dtype=np.float32).reshape(count, self.dim)
                records.append(
                    WALRecord(
                        lsn=int(lsn),
                        op_type=int(op_type),
                        ids=ids.copy(),
                        vectors=np.ascontiguousarray(vectors.copy(), dtype=np.float32),
                    )
                )
        return records

    def truncate_before(self, lsn: int) -> None:
        records = [record for record in self.recover() if record.lsn >= lsn]
        if self._file is not None:
            self._file.close()
            self._file = None

        tmp_path = f"{self.file_path}.tmp"
        with open(tmp_path, "wb") as f:
            for record in records:
                f.write(
                    self._serialize_record(
                        record.lsn,
                        record.op_type,
                        record.ids,
                        record.vectors,
                    )
                )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.file_path)
        self._write_meta_lsn()
        self._file = open(self.file_path, "ab")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def last_lsn(self) -> int:
        return self._last_lsn

    @staticmethod
    def _resolve_file_path(path: str) -> str:
        _, ext = os.path.splitext(path)
        if ext:
            return path
        return os.path.join(path, "wal.log")

    def _read_meta_lsn(self) -> int:
        try:
            with open(self.meta_path) as f:
                return int(f.read().strip() or "0")
        except FileNotFoundError:
            return 0

    def _write_meta_lsn(self) -> None:
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "w") as f:
            f.write(str(self._last_lsn))

    def _prepare_record_arrays(
        self, ids: np.ndarray, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.ascontiguousarray(ids.astype(np.uint64))
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape [N, {self.dim}].")
        if len(ids) != len(vectors):
            raise ValueError("len(ids) != len(vectors)")
        return ids, vectors

    def _serialize_record(
        self, lsn: int, op_type: int, ids: np.ndarray, vectors: np.ndarray
    ) -> bytes:
        ids, vectors = self._prepare_record_arrays(ids, vectors)
        header = _HEADER.pack(int(lsn), int(op_type), len(ids))
        body = ids.tobytes(order="C") + vectors.tobytes(order="C")
        checksum = _CHECKSUM.pack(zlib.crc32(header + body) & 0xFFFFFFFF)
        return header + body + checksum
