"""모델 의존 파일과 wirlwind 참조 자산의 무결성 검사."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from goalkeeper_mjlab import PACKAGE_ROOT


def validate_model_assets(xml_path: Path) -> tuple[Path, ...]:
    """XML이 실제로 참조하는 mesh/texture 파일이 비어 있거나 누락되면 중단한다."""
    if not xml_path.is_file() or xml_path.stat().st_size == 0:
        raise FileNotFoundError(f"Model XML is missing or empty: {xml_path}")
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    paths = [xml_path]
    for kind, directory in (("mesh", "meshdir"), ("texture", "texturedir")):
        base = compiler.get(directory, "") if compiler is not None else ""
        for asset in root.findall(f"./asset/{kind}"):
            if asset.get("file"):
                paths.append(xml_path.parent / base / asset.attrib["file"])
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Referenced asset is missing or empty: {path}")
    return tuple(paths)


def verify_source_assets() -> dict:
    """wirlwind 참조 저장소 폴더 없이, 복사 당시 기록한 SHA-256과 비교한다."""
    manifest = json.loads((PACKAGE_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for relative, record in manifest["files"].items():
        # files의 키(robots/k1/... 등)는 설치된 이 패키지 안의 상대경로다.
        # record의 source_path는 wirlwind GitHub 저장소에서 어디에 있었는지 적은 출처 기록이다.
        # 그 경로를 열거나 이웃 wirlwind 참조 저장소 폴더를 찾지 않는다. 서버에는 이 패키지만 있으면 된다.
        path = PACKAGE_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            mismatches.append(relative)
    return {
        "source_commit": manifest["source_commit"],
        "file_count": len(manifest["files"]),
        "mismatches": mismatches,
    }


def load_model_spec(xml_path: Path):
    """메시 바이트까지 VFS에 넣어, wirlwind 참조 저장소 폴더 없이도 내보낸 장면을 다시 열 수 있게 한다."""
    import mujoco
    paths = validate_model_assets(xml_path)
    # from_file만 사용하면 현재 장면은 열려도 to_zip에 메시 파일이 빠진다.
    assets = {os.path.relpath(path, xml_path.parent): path.read_bytes() for path in paths[1:]}
    return mujoco.MjSpec.from_string(xml_path.read_text(encoding="utf-8"), assets=assets)
