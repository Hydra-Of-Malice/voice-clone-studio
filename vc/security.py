"""Encryption at rest for voice profiles (biometric data). Envelope scheme: one random data key per
profile, wrapped by a master key kept outside the profile folder. Reference clips and embeddings are
stored only in encrypted form and decrypted to memory / a short-lived temp file at generation time."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet


class KeyStore:
    def __init__(self, data_dir: Path):
        self.master_path = data_dir / "master.key"
        if not self.master_path.exists():
            self.master_path.parent.mkdir(parents=True, exist_ok=True)
            self.master_path.write_bytes(Fernet.generate_key())
            try:
                os.chmod(self.master_path, stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass
        self._master = Fernet(self.master_path.read_bytes())

    def new_profile_key(self) -> tuple[bytes, bytes]:
        """Returns (plaintext key, wrapped key)."""
        key = Fernet.generate_key()
        return key, self._master.encrypt(key)

    def unwrap(self, wrapped: bytes) -> bytes:
        return self._master.decrypt(wrapped)


class ProfileCipher:
    def __init__(self, key: bytes):
        self._f = Fernet(key)

    def encrypt_file(self, src: Path, dst: Path) -> None:
        dst.write_bytes(self._f.encrypt(src.read_bytes()))

    def encrypt_bytes(self, data: bytes) -> bytes:
        return self._f.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        return self._f.decrypt(data)
