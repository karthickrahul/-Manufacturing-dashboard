"""Convert NEU-DET (Pascal VOC) to YOLO format and make a stratified 70/15/15 split.

Usage: python src/prepare_data.py
"""
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

from sklearn.model_selection import train_test_split

from common import CLASSES, DATA_RAW, DS, SEED, cls_of

IMG_DIRS = ["IMAGES", "Validation_Images"]          # the source author moved 30 files to Validation_*
XML_DIRS = ["ANNOTATIONS", "Validation_Annotations"]


def voc_to_yolo(xml_path):
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    images, xmls = {}, {}
    for d in IMG_DIRS:
        for p in (DATA_RAW / d).glob("*.jpg"):
            images[p.stem] = p
    for d in XML_DIRS:
        for p in (DATA_RAW / d).glob("*.xml"):
            xmls[p.stem] = p
    stems = sorted(set(images) & set(xmls))
    print(f"{len(stems)} image+annotation pairs (expected 1800)")
    assert len(stems) == 1800, "Dataset incomplete - run scripts/get_data.sh"

    found = Counter()
    for s in stems:
        for o in ET.parse(xmls[s]).getroot().findall("object"):
            found[o.find("name").text.strip()] += 1
    print("Boxes per class in XML:", dict(found))
    assert set(found) <= set(CLASSES), f"Unexpected class names: {set(found) - set(CLASSES)}"

    strat = [cls_of(s) for s in stems]
    train, rest = train_test_split(stems, test_size=0.30, stratify=strat, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=[cls_of(s) for s in rest], random_state=SEED)

    if DS.exists():
        shutil.rmtree(DS)
    for split, items in {"train": train, "val": val, "test": test}.items():
        (DS / "images" / split).mkdir(parents=True)
        (DS / "labels" / split).mkdir(parents=True)
        for s in items:
            shutil.copy(images[s], DS / "images" / split / images[s].name)
            (DS / "labels" / split / f"{s}.txt").write_text("\n".join(voc_to_yolo(xmls[s])) + "\n")
        print(split, len(items), dict(sorted(Counter(cls_of(s) for s in items).items())))

    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    (DS / "data.yaml").write_text(
        f"path: {DS.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}")
    print("Wrote", DS / "data.yaml")


if __name__ == "__main__":
    main()
