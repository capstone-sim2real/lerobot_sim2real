#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_name=localhost/so101-report-tex:ubuntu24.04
if ! podman image exists "$image_name"; then
    podman build -t "$image_name" -f "$report_dir/Containerfile" "$report_dir"
fi
podman run --rm --security-opt label=disable \
    -v "$report_dir:/work:rw" "$image_name" \
    latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir=build final_report.tex
cp "$report_dir/build/final_report.pdf" "$report_dir/final_report.pdf"
# Keep the former revised filename compatible with the single submission source.
cp "$report_dir/build/final_report.pdf" "$report_dir/final_report_revised.pdf"
