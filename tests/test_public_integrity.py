from pathlib import Path
import pytest
from reproduction_support import digest, register_local_files, validate_reproduction_file, validate_source_manifest


def test_generated_inputs_are_checked_after_registration(tmp_path):
    (tmp_path / 'SOURCE_MANIFEST.sha256').write_text('')
    p = tmp_path / 'input.csv'
    p.write_text('synthetic,1\n')
    register_local_files(tmp_path, [p], stage='prepare')
    assert validate_reproduction_file(p, 'historical-digest', tmp_path) == digest(p)
    p.write_text('synthetic,2\n')
    with pytest.raises(RuntimeError, match='changed'):
        validate_reproduction_file(p, 'historical-digest', tmp_path)


def test_unregistered_different_input_is_rejected(tmp_path):
    (tmp_path / 'SOURCE_MANIFEST.sha256').write_text('')
    p = tmp_path / 'input.csv'
    p.write_text('synthetic,1\n')
    with pytest.raises(RuntimeError, match='changed'):
        validate_reproduction_file(p, 'historical-digest', tmp_path)


def test_modified_public_source_is_rejected(tmp_path):
    p = tmp_path / 'kernel.py'
    p.write_text('x = 1\n')
    (tmp_path / 'SOURCE_MANIFEST.sha256').write_text(f'{digest(p)}  kernel.py\n')
    validate_source_manifest(tmp_path)
    p.write_text('x = 2\n')
    with pytest.raises(RuntimeError, match='manifest mismatch'):
        validate_source_manifest(tmp_path)
