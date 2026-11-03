"""이 파일을 실행하면 K1 골키퍼 장면을 MuJoCo viewer로 연다."""

from pathlib import Path
import sys

# goalkeeper_mjlab 자체를 설치하지 않아도 IDE/프로젝트 외부에서 소스를 찾도록 한다.
# 단, MJLab·MuJoCo 같은 외부 의존성은 이 파일을 실행하는 Python 가상환경에 설치되어야 한다.
# 다른 프로젝트의 Python에 필요한 패키지가 없다면, 이 경로 추가만으로 해결되지는 않는다.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from goalkeeper_mjlab.scripts.inspect_scene import main


if __name__ == "__main__":
    raise SystemExit(main())
