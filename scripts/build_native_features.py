"""Build once under resource_guard; record exact compiler, source and binary identity."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from training.native_candidate_features import ABI, FLAGS
    from resource_guard import now, write_json, operation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = Path('/usr/bin/cc').resolve(strict=True)
    binary = output/'candidate_features.so'
    source = ROOT/'training/native_candidate_features.c'
    wrapper = ROOT/'training/native_candidate_features.py'
    version = subprocess.run([str(compiler), '--version'], check=True, capture_output=True, text=True).stdout
    command = [str(compiler), *FLAGS, str(source), '-o', str(binary)]
    subprocess.run(command, check=True)
    manifest = dict(at=now(), abi=ABI, compiler=str(compiler), compiler_version=version,
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(), flags=FLAGS,
        platform=platform.platform(), machine=platform.machine(), command=command,
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(), binary=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    write_json(output/'manifest.json', manifest)
    operation('隔离C候选特征原型编译完成，仅固定来源/ABI/编译器/二进制身份，未接训练', output/'manifest.json')
    print(json.dumps({'binary_sha256': manifest['binary_sha256'], 'abi': ABI}))


if __name__ == '__main__':
    main()
