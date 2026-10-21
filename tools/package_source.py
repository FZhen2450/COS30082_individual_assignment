"""Build the assignment source ZIP without images, environments or weights."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
output = root / "submission" / "COS30082_source.zip"
output.parent.mkdir(exist_ok=True)
files = [root / name for name in [
    "Bird_Classification_Local_GPU.ipynb", "birdlab.py", "bird_visuals.py", "app.py", "requirements.txt",
    "setup_gpu.ps1", "README.md", "ASSIGNMENT_GUIDE.md", "PROGRESS.md",
    ".gitignore",
    "train.txt", "test.txt",
]]
files += sorted((root / "tools").glob("*.py"))
files += sorted((root / "artifacts").rglob("*.csv"))
files += sorted((root / "artifacts").rglob("*.png"))
files += sorted((root / "artifacts").rglob("*.json"))
files += sorted((root / "deployment").glob("*.json"))
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for path in files:
        if path.is_file():
            archive.write(path, path.relative_to(root))
print(output)
print("Source and available experiment results included; weights/images excluded.")
