from pathlib import Path


def test_requirements_include_api_runtime_dependencies():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert any(line.startswith("fastapi") for line in requirements)
    assert any(line.startswith("uvicorn") for line in requirements)
