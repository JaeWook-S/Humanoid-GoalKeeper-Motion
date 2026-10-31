"""Goalkeeper_mjlab 하나만 복사한 서버 조건에서 경로·출처 기록의 독립성을 검사한다."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PortabilityTests(unittest.TestCase):
    def test_standalone_copy_runs_offline_without_original_repositories(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="goalkeeper-portability-") as directory:
            root = Path(directory)
            clone = root / "only_repository"
            shutil.copytree(project, clone, ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "*.pyc", "*.egg-info", "dist", "build",
            ))
            package = clone / "src/goalkeeper_mjlab"
            # 출처 기록을 일부러 열 수 없는 값으로 바꿔도, 실제 패키지 파일만으로 동작해야 한다.
            manifest_path = package / "asset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for record in manifest["files"].values():
                record["source_path"] = "/never-mounted/reference-repository/missing.xml"
            manifest["source_repository"] = "https://example.invalid/source"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            reference_path = package / "robots/k1/motor_reference.json"
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            for record in reference["official_files"].values():
                record["url"] = "https://example.invalid/not-available.py"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")

            unrelated = root / "unrelated_working_directory"
            unrelated.mkdir()
            blocked = [project / "src", project / "main.py",
                       project.parent / "booster_mjlab", project.parent / "inha_mjlab"]
            # 단순히 복사만 하는 데서 끝내지 않는다. 원래 소스 읽기와 네트워크 접속을 실제로 거부한다.
            guard = """
import json, os, pathlib, runpy, sys
blocked = [pathlib.Path(p).resolve() for p in json.loads(sys.argv[1])]
entry = sys.argv[2]
def audit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('Standalone model execution must not fetch source metadata')
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(base) for base in blocked):
            raise AssertionError('Standalone execution accessed an original repository: ' + str(path))
sys.addaudithook(audit)
sys.argv = [entry, *sys.argv[3:]]
runpy.run_path(entry, run_name='__main__')
"""
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for args in (["--stage", "1"], ["--check-motors", "--hold-seconds", "1"]):
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, "-c", guard, json.dumps([str(p) for p in blocked]),
                         str(clone / "main.py"), "--headless", *args],
                        cwd=unrelated, env=env, text=True, capture_output=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertTrue(report["passed"], report.get("errors"))
                    self.assertEqual(report["source_assets"]["file_count"], 31)


if __name__ == "__main__":
    unittest.main()
