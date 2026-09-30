"""Collect every data file the 261001 paper rests on into archive/<tag>/ with a sha256 manifest,
then tar it. The tarball stays out of git (archive/ is ignored); the manifest is copied to
docs/data_manifest_<tag>.json and committed, so the archive can be verified or rebuilt later.

Run from the repository root:  python3 proposed_src/util/archive_paper_data.py --tag 261001
"""
import argparse, glob, hashlib, json, os, shutil, tarfile, time

GROUPS = {
	'evaluation instances (Tables 3-12, all figures)': ['instances/v22test_10x25/*.json', 'instances/v22test_18x50/*.json', 'instances/v22test_21x60/*.json',
	                                                    'instances/v24test_21x60_100/*.json', 'instances/v24test_24x70/*.json', 'instances/v24test_28x80/*.json'],
	'Stage-1 training/test states 10x25 (Sec. 4.4, Table 9, Fig. 8)': ['runs/v22/cond_10x25_train.json', 'runs/v22/cond_10x25_test.json', 'runs/v22/cond_18x50_train.json', 'runs/v22/cond_18x50_test.json'],
	'Stage-1 checkpoint and depth evaluation': ['runs/v22/model_n25/*'],
	'deployment-range states and M1/M2 checkpoints (Sec. 5.5.2, Fig. 9)': ['runs/v24/cond_*.json', 'runs/v24/M1/*', 'runs/v24/M2/*', 'runs/v24/train_*.json', 'runs/v24/where_M012.json'],
	'Stage-2 pools, critic data, checkpoints, logs (Sec. 4.5, Fig. 6a)': ['runs/v24/pool_*.json', 'runs/v24/r1_*.json', 'runs/v24/critic_frozen/*', 'runs/v24/critic_18x50/*', 'runs/v24/r2/*', 'runs/v24/r2_seed1/*', 'runs/v24/r2_18x50/*', 'runs/v24/r0_*.json', 'runs/v24/pv_root_diag.log'],
	'search results v24 (Tables 3-8, 10; Fig. 6b, 7)': ['runs/v24/summary_*.json', 'runs/v24/*_ahloff_*/inst-*_r.json', 'runs/v24/*_ahlon_*/inst-*_r.json', 'runs/v24/*.log'],
	'search results v23 (cascade Tables 11-12, 30-instance sets)': ['runs/v23/aggregate.json', 'runs/v23/*_S?_ahl*/inst-*_r.json', 'runs/v23/*.log'],
	'revision analyses (classical solvers, control training, lattice, depth curves, ablation rescoring)': ['runs/revision/*.json', 'runs/revision/*.log', 'runs/revision/ctrl_root/*', 'runs/revision/ctrl_intree/*'],
	'control training data (root vs in-tree)': ['runs/revision/ctrl_intree_train.json', 'runs/revision/ctrl_root_train.json'],
}


def sha256(p):
	h = hashlib.sha256()
	with open(p, 'rb') as f:
		for chunk in iter(lambda: f.read(1 << 20), b''):
			h.update(chunk)
	return h.hexdigest()


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--tag', required=True)
	ap.add_argument('--out', default='archive')
	a = ap.parse_args()
	root = os.path.join(a.out, f'{a.tag}_paper_data')
	os.makedirs(root, exist_ok=True)
	manifest = dict(tag=a.tag, created=time.strftime('%Y-%m-%d %H:%M:%S'), groups={}, files=[])
	total = 0
	for group, pats in GROUPS.items():
		fs = sorted({f for p in pats for f in glob.glob(p) if os.path.isfile(f)})
		manifest['groups'][group] = dict(patterns=pats, count=len(fs))
		for f in fs:
			dst = os.path.join(root, f)
			os.makedirs(os.path.dirname(dst), exist_ok=True)
			shutil.copy2(f, dst)
			sz = os.path.getsize(f); total += sz
			manifest['files'].append(dict(path=f, bytes=sz, sha256=sha256(f), group=group))
		print(f'{len(fs):6d} files  {group}', flush=True)
	manifest['total_bytes'] = total
	json.dump(manifest, open(os.path.join(root, 'manifest.json'), 'w'), indent=1)
	json.dump(manifest, open(f'docs/data_manifest_{a.tag}.json', 'w'), indent=1)
	tar = os.path.join(a.out, f'{a.tag}_paper_data.tar.gz')
	with tarfile.open(tar, 'w:gz') as t:
		t.add(root, arcname=os.path.basename(root))
	print(f'files {len(manifest["files"])}  {total/1e6:.1f} MB  -> {tar} ({os.path.getsize(tar)/1e6:.1f} MB)')


if __name__ == '__main__':
	main()
