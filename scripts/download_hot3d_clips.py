"""
Fetch missing HOT3D-Clips (Aria train) from bop-benchmark/hot3d on Hugging Face.

HOT3D-Clips is ungated there and carries the same annotations as the full HOT3D
release, which needs a credentialed manifest from projectaria.com. The
authoritative clip list is clip_splits.json in that repo: train/Aria is ids
1849..3364 (1516 clips).

Uses huggingface_hub directly rather than the CLI: `huggingface-cli` is
deprecated and now prints `hf` usage instead of downloading, which looks like
success to a shell script -- the first attempt at this "completed" having
fetched nothing, and only a size check caught it.

Licence: HOT3D is non-commercial research use.
"""
import argparse
import os
import sys

MIN_BYTES = 10_000_000        # a real clip is ~100 MB; smaller means truncated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lo', type=int, default=2849)
    p.add_argument('--hi', type=int, default=3364)
    p.add_argument('--dest', default='datasets/hot3d_clips')
    p.add_argument('--retries', type=int, default=3)
    a = p.parse_args()

    from huggingface_hub import hf_hub_download
    ok = skip = fail = 0
    total = a.hi - a.lo + 1
    for i in range(a.lo, a.hi + 1):
        name = f'clip-{i:06d}.tar'
        local = os.path.join(a.dest, 'train_aria', name)
        if os.path.exists(local) and os.path.getsize(local) > MIN_BYTES:
            skip += 1
            continue
        for attempt in range(1, a.retries + 1):
            try:
                hf_hub_download('bop-benchmark/hot3d', f'train_aria/{name}',
                                repo_type='dataset', local_dir=a.dest)
                if os.path.getsize(local) > MIN_BYTES:
                    ok += 1
                    break
                print(f'[FAIL] {name}: only {os.path.getsize(local)} bytes', flush=True)
            except Exception as e:
                if attempt == a.retries:
                    print(f'[FAIL] {name}: {type(e).__name__} {str(e)[:90]}', flush=True)
        else:
            fail += 1
        done = ok + skip + fail
        if done % 25 == 0:
            print(f'  {done}/{total}  ok={ok} skip={skip} fail={fail}', flush=True)
    print(f'ALL DONE  ok={ok} skip={skip} fail={fail}', flush=True)
    have = len([f for f in os.listdir(os.path.join(a.dest, 'train_aria'))
                if f.endswith('.tar')])
    print(f'total clips now: {have}')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
