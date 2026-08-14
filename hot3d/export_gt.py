"""
Export HOT3D ground truth for a chosen set of sequences.

Run from the `hot3d/` directory (the exporter resolves `dataset/assets` and
`mano_v1_2/models/` relative to the working directory).

    # Unchanged default: re-export the 27 evaluation sequences, write val.json
    python export_gt.py

    # Every downloaded sequence that is NOT in the evaluation set -> train.json
    python export_gt.py --split train --all-available

    # An explicit list, or a list from a file (.json / .pkl / .txt)
    python export_gt.py --split train --sequences P0001_a68492d5 P0002_2ea9af5b
    python export_gt.py --split train --sequence-file my_train_seqs.txt

The written `<split>.json` lists only the sequences that exported successfully,
so downstream preprocessing never points at a missing directory.
"""
import argparse
import json
import os
import pickle
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# viewer_export_gt imports rerun and the projectaria data loaders at module
# level, so it is imported lazily: --help, --dry-run and split selection then
# work without `uv sync --extra hot3d`.

# The evaluation split used by scripts/scripts_eval/eval_hawor_hot3d.py. Kept
# here as the single source of truth so a training split can be checked against it.
VAL_SEQUENCES = [
    'P0001_a68492d5', 'P0001_9b6feab7', 'P0014_8254f925', 'P0011_76ea6d47',
    'P0014_84ea2dcc', 'P0001_8d136980', 'P0012_476bae57', 'P0012_130a66e1',
    'P0014_24cb3bf0', 'P0010_1c9fe708', 'P0002_2ea9af5b', 'P0011_11475e24',
    'P0010_0ecbf39f', 'P0010_160e551c', 'P0015_42b8b389', 'P0012_915e71c6',
    'P0002_65085bfc', 'P0011_47878e48', 'P0011_cee8fe4f', 'P0002_016222d1',
    'P0012_d85e10f6', 'P0012_119de519', 'P0010_41c4c626', 'P0012_f7e3880b',
    'P0009_02511c2f', 'P0011_72efb935', 'P0010_924e574e',
]

# Directory names under --dataset-root that are not sequences.
NON_SEQUENCE_DIRS = {'assets'}


def load_set(set_file):
    """Read a sequence list from a .pkl, .json or newline-delimited .txt file."""
    ext = os.path.splitext(set_file)[1].lower()
    if ext == '.pkl':
        with open(set_file, 'rb') as f:
            return list(pickle.load(f))
    if ext == '.json':
        with open(set_file, 'r') as f:
            return list(json.load(f))
    with open(set_file, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def discover_sequences(dataset_root):
    """Every sequence directory present under dataset_root."""
    if not os.path.isdir(dataset_root):
        raise SystemExit(f'--dataset-root {dataset_root} does not exist. '
                         'Download the sequences first (see README).')
    names = []
    for name in sorted(os.listdir(dataset_root)):
        if name.startswith('.') or name in NON_SEQUENCE_DIRS:
            continue
        if os.path.isdir(os.path.join(dataset_root, name)):
            names.append(name)
    return names


def is_exported(export_folder):
    """A sequence counts as exported once its ground truth blob is on disk."""
    return os.path.exists(os.path.join(export_folder, 'anno.pth'))


def select_sequences(args):
    if args.sequences:
        selected, source = list(args.sequences), 'command line'
    elif args.sequence_file:
        selected, source = load_set(args.sequence_file), args.sequence_file
    elif args.all_available:
        selected, source = discover_sequences(args.dataset_root), f'{args.dataset_root}/'
    else:
        selected, source = list(VAL_SEQUENCES), 'built-in evaluation split'
    print(f'{len(selected)} sequences requested from {source}')

    # Guard against train/test contamination: silently exporting evaluation
    # sequences into a training split would inflate every number you report.
    if args.split != 'val' and not args.allow_val_overlap:
        overlap = [s for s in selected if s in VAL_SEQUENCES]
        if overlap:
            selected = [s for s in selected if s not in VAL_SEQUENCES]
            print(f'Excluded {len(overlap)} evaluation sequence(s) from split '
                  f"'{args.split}': {', '.join(overlap[:5])}"
                  f"{'...' if len(overlap) > 5 else ''}")
            print('Pass --allow-val-overlap to keep them (you almost certainly do not want this).')

    missing = [s for s in selected if not os.path.isdir(os.path.join(args.dataset_root, s))]
    if missing:
        print(f'WARNING: {len(missing)} requested sequence(s) are not downloaded and '
              f"will be skipped: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
        selected = [s for s in selected if s not in set(missing)]
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', type=str, default='val',
                        help="Name of the split; also the output manifest name (<split>.json).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sequences', type=str, nargs='+',
                       help='Explicit sequence names.')
    group.add_argument('--sequence-file', type=str,
                       help='File listing sequence names (.json, .pkl or .txt).')
    group.add_argument('--all-available', action='store_true',
                       help='Use every sequence directory found under --dataset-root.')
    parser.add_argument('--dataset-root', type=str, default='dataset',
                        help='Where the downloaded sequences live.')
    parser.add_argument('--output-root', type=str, default='hot3d_dataset_export',
                        help='Where the exported ground truth is written.')
    parser.add_argument('--mano-model-folder', type=str, default='mano_v1_2/models/')
    parser.add_argument('--start-frame', type=int, default=20)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip sequences that already have anno.pth. Use to resume.')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Keep going if a sequence fails, and report failures at the end.')
    parser.add_argument('--allow-val-overlap', action='store_true',
                        help='Permit evaluation sequences in a non-val split.')
    parser.add_argument('--dry-run', action='store_true',
                        help='List what would be exported, then exit.')
    return parser.parse_args()


def main():
    args = parse_args()
    selected = select_sequences(args)
    if not selected:
        raise SystemExit('Nothing to export.')

    manifest_path = os.path.join(args.output_root, f'{args.split}.json')
    if args.dry_run:
        print(f'\n[dry run] would export {len(selected)} sequence(s) to {args.output_root}/')
        for name in selected:
            print(f'  {name}')
        print(f'[dry run] would write {manifest_path}')
        return

    os.makedirs(args.output_root, exist_ok=True)
    object_library_folder = os.path.join(args.dataset_root, 'assets')

    exported, skipped, failed = [], [], []
    for i, name in enumerate(selected, 1):
        sequence_folder = os.path.join(args.dataset_root, name)
        export_folder = os.path.join(args.output_root, name)

        if args.skip_existing and is_exported(export_folder):
            print(f'[{i}/{len(selected)}] {name}: already exported, skipping')
            skipped.append(name)
            exported.append(name)
            continue

        print(f'[{i}/{len(selected)}] exporting {name} ...')
        try:
            # Imported here so a resume that exports nothing (--skip-existing)
            # does not need the HOT3D toolkit installed.
            from viewer_export_gt import export_gt
            export_gt(sequence_folder,
                      start_frame=args.start_frame,
                      export_gt_folder=export_folder,
                      object_library_folder=object_library_folder,
                      mano_model_folder=args.mano_model_folder)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            if not args.continue_on_error:
                # Still record what succeeded, so a long run is not lost.
                _write_manifest(manifest_path, exported)
                raise SystemExit(
                    f'{name} failed. Fix it, or re-run with --continue-on-error '
                    f'--skip-existing to resume.')
            continue
        exported.append(name)

    _write_manifest(manifest_path, exported)
    print(f'\nexported {len(exported) - len(skipped)}, skipped {len(skipped)}, '
          f'failed {len(failed)}')
    if failed:
        print('failed sequences: ' + ', '.join(failed))


def _write_manifest(path, sequences):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(sequences, f, indent=1)
    print(f'wrote {path} with {len(sequences)} sequence(s)')


if __name__ == '__main__':
    main()
