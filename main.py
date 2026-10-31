"""이 파일을 실행하면 K1 골키퍼 장면을 MuJoCo viewer로 연다."""

from pathlib import Path
import sys

# 패키지 설치 없이도 IDE의 실행 버튼이나 프로젝트 외부 경로에서 실행할 수 있게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from goalkeeper_mjlab.scripts.inspect_scene import main


if __name__ == "__main__":
    raise SystemExit(main())
