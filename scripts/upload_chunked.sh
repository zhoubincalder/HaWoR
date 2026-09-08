#!/usr/bin/env bash
# Upload an archive that is too large for lakectl's multipart in one object.
#
# lakectl 1.80.0 against lakeFS 1.79.0 refused the 43GB DexYCB archive with
#     parameter "partNumber" in path has an error: number must be at most 1000
# having asked for part 3107, i.e. a part size of ~14.5MB. That caps a single
# object at roughly 14GB. (Objects of 17.1GB and 19.5GB did upload on an
# earlier occasion, so the part size is evidently not always the same -- this
# script does not try to explain that, it just stops depending on it.)
#
# So an oversized archive is split into fixed-size chunks, each uploaded as its
# own object with its own sha256, plus a .parts sidecar recording the chunk
# count and the sha256 of the WHOLE archive. Restore is:
#
#     lakectl fs download <...>.tar.zst.part00 ...   # every part
#     cat X.tar.zst.part* > X.tar.zst
#     sha256sum -c   # against the whole-file sha in the .parts sidecar
#     tar -I zstd -xf X.tar.zst
#
# The whole-file sha is what makes this safe: a missing or truncated chunk
# changes it, so a partial restore cannot pass as complete. That is the same
# failure this corpus already suffered once, at a coarser grain.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-lakefs://calder-dev}"
BRANCH="${BRANCH:-hawor-stage-a}"
PREFIX="${PREFIX:-datasets/hawor/stage-a}"
CHUNK="${CHUNK:-10000000000}"   # 10 GB: comfortably under a ~14GB ceiling
a="${1:?usage: upload_chunked.sh <archive.tar.zst>}"

[ -f "$a" ] || { echo "no such file: $a"; exit 1; }
n=$(basename "$a")
sz=$(stat -c%s "$a")
whole=$(cat "$a.sha256" 2>/dev/null | tr -dc '0-9a-f' | head -c 64)
if [ -z "$whole" ]; then
  echo "[sha ] computing whole-file sha256"
  whole=$(sha256sum "$a" | awk '{print $1}')
  echo "$whole" > "$a.sha256"
fi

work="$(dirname "$a")/_chunks/$n"
mkdir -p "$work"
have=$(ls "$work"/part?? 2>/dev/null | wc -l)
if [ "$have" -eq 0 ]; then
  echo "[split] $n $(numfmt --to=iec "$sz") into $(( (sz + CHUNK - 1) / CHUNK )) chunks"
  split -b "$CHUNK" -d -a 2 "$a" "$work/part" || { echo "[FAIL] split"; exit 1; }
fi
parts=$(ls "$work"/part?? | wc -l)
# A split that produced the wrong total is worse than no backup at all.
tot=$(cat "$work"/part?? | wc -c)
if [ "$tot" -ne "$sz" ]; then
  echo "[FAIL] chunks total $tot bytes, archive is $sz"; exit 1
fi
echo "[ok   ] $parts chunks, $tot bytes = archive size"

fail=0
for p in "$work"/part??; do
  suf=$(basename "$p")            # partNN
  obj="$REPO/$BRANCH/$PREFIX/$n.$suf"
  psha=$(sha256sum "$p" | awk '{print $1}')
  rsha=$(lakectl fs cat "$obj.sha256" 2>/dev/null | tr -dc '0-9a-f' | head -c 64)
  if [ "$psha" = "$rsha" ]; then
    echo "[skip ] $n.$suf (sha matches)"; continue
  fi
  echo "[put  ] $n.$suf $(numfmt --to=iec "$(stat -c%s "$p")")"
  if lakectl fs upload -s "$p" "$obj"; then
    echo "$psha" > "$p.sha256"
    lakectl fs upload -s "$p.sha256" "$obj.sha256" >/dev/null
  else
    echo "[FAIL ] $n.$suf"; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "FAILED: some chunks did not upload"; exit 1; }

# The sidecar that makes a restore verifiable.
meta="$work/parts.json"
cat > "$meta" <<JSON
{
  "archive": "$n",
  "bytes": $sz,
  "sha256": "$whole",
  "chunks": $parts,
  "chunk_bytes": $CHUNK,
  "restore": "cat $n.part* > $n && sha256sum -c (expect $whole) && tar -I zstd -xf $n"
}
JSON
lakectl fs upload -s "$meta" "$REPO/$BRANCH/$PREFIX/$n.parts" >/dev/null \
  || { echo "[FAIL] parts sidecar"; exit 1; }
echo "[ok   ] uploaded $parts chunks + $n.parts sidecar"
rm -f "$work"/part??            # chunks are reproducible from the archive
echo "ALL DONE"
