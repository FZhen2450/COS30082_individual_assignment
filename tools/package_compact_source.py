"""Package the separate compact study without overwriting the detailed ZIP."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
output = root / "submission/COS30082_compact_source.zip"
output.parent.mkdir(exist_ok=True)
files = [root / name for name in [
    "Bird_Classification_Compact_GPU.ipynb", "birdlab_compact.py", "birdlab.py",
    "app_compact.py", "COMPACT_GUIDE.md", "ASSIGNMENT_GUIDE.md", "requirements.txt",
    "setup_gpu.ps1", "train.txt", "test.txt", "tools/build_compact_notebook.py",
    "tools/verify_compact.py", "tools/package_compact_source.py"]]
for pattern in ["*.csv", "*.json", "*.png"]:
    files += sorted((root / "artifacts_compact_v1").rglob(pattern))
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for path in files:
        if path.is_file():
            archive.write(path, path.relative_to(root))
print(output)
