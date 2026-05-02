"""Unit tests for the binary write-ahead log."""

import os

import numpy as np

from vecscaledb.storage.wal import OP_INSERT, WAL


DIM = 8


def test_wal_append_reopen_and_recover(tmp_path):
    wal = WAL(str(tmp_path / "wal"), DIM)
    wal.open()

    for batch in range(3):
        ids = np.array([batch * 10, batch * 10 + 1], dtype=np.int64)
        vectors = np.full((2, DIM), batch, dtype=np.float32)
        assert wal.append(OP_INSERT, ids, vectors) == batch + 1
    wal.close()

    recovered_wal = WAL(str(tmp_path / "wal"), DIM)
    recovered_wal.open()
    records = recovered_wal.recover()
    recovered_wal.close()

    assert [record.lsn for record in records] == [1, 2, 3]
    assert [record.op_type for record in records] == [OP_INSERT] * 3
    np.testing.assert_array_equal(records[2].ids, np.array([20, 21]))
    np.testing.assert_allclose(records[2].vectors, np.full((2, DIM), 2))


def test_wal_recovery_skips_corrupted_trailing_record(tmp_path):
    wal = WAL(str(tmp_path / "wal"), DIM)
    wal.open()
    for batch in range(3):
        ids = np.array([batch], dtype=np.int64)
        vectors = np.full((1, DIM), batch, dtype=np.float32)
        wal.append(OP_INSERT, ids, vectors)
    wal.close()

    with open(wal.file_path, "r+b") as f:
        f.seek(-4, os.SEEK_END)
        f.write(b"\x00\x00\x00\x00")

    recovered = WAL(str(tmp_path / "wal"), DIM).recover()

    assert [record.lsn for record in recovered] == [1, 2]
