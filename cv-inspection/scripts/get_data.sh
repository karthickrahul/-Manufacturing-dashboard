#!/usr/bin/env bash
# Download the NEU-DET images + Pascal VOC annotations into data/raw/.
#
# The NEU surface defect database (Song & Yan, Northeastern University) is not
# redistributed in this repository. This script fetches a public GitHub copy at a
# pinned commit so results are reproducible. Check the dataset's terms of use on the
# original page before using it beyond research / coursework:
#   http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
set -euo pipefail

REPO="https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection"
COMMIT="351bee21b56bbf97a1e1d18262bceb202eb7ef41"
DEST="${1:-data/raw}"

if [ -d "$DEST/IMAGES" ] && [ -d "$DEST/ANNOTATIONS" ]; then
  echo "Dataset already present in $DEST"
  exit 0
fi

mkdir -p "$DEST"
tmp="$(mktemp -d)"
git init --quiet "$tmp/src"
git -C "$tmp/src" fetch --quiet --depth 1 "$REPO" "$COMMIT"
git -C "$tmp/src" checkout --quiet FETCH_HEAD
cp -r "$tmp/src/IMAGES" "$tmp/src/ANNOTATIONS" "$tmp/src/Validation_Images" "$tmp/src/Validation_Annotations" "$DEST/"
rm -rf "$tmp"
echo "Done: $(find "$DEST" -iname '*.jpg' | wc -l) images, $(find "$DEST" -iname '*.xml' | wc -l) annotation files"
