from __future__ import annotations

import pytest

from parkinsons_voice.data.schema import ManifestValidationError, shareable_manifest
from parkinsons_voice.data.uci489 import load_uci489


def test_manifest_uses_opaque_unique_keys_and_hides_locator(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)

    manifest = load_uci489(source).manifest

    assert manifest["sample_key"].is_unique
    assert manifest["participant_key"].nunique() == 20
    assert manifest["participant_key"].str.startswith("p_").all()
    assert "source_locator" not in shareable_manifest(manifest).columns


def test_manifest_rejects_duplicate_sample_key(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)
    manifest = load_uci489(source).manifest
    manifest.loc[1, "sample_key"] = manifest.loc[0, "sample_key"]

    with pytest.raises(ManifestValidationError, match="duplicate sample_key"):
        shareable_manifest(manifest)
