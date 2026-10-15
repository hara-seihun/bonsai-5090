#!/usr/bin/env python3
"""Own a server's lifetime while running a declared paired task panel.

Run under flock /tmp/binary-gpu.lock. Commands and environment are explicit JSON.
A quick 32+32 panel is a scoped task panel, not the full quality gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse


def listening_inodes(port):
    found = set()
    for name in ('tcp', 'tcp6'):
        for line in Path('/proc/self/net/' + name).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                found.add(fields[9])
    return found


def owns_listener(pid, port):
    listeners = listening_inodes(port)
    if not listeners:
        return False
    owned = set()
    try:
        for fd in Path(f'/proc/{pid}/fd').iterdir():
            try:
                link = os.readlink(fd)
            except FileNotFoundError:
                continue
            if link.startswith('socket:[') and link.endswith(']'):
                owned.add(link[8:-1])
    except FileNotFoundError:
        return False
    if not listeners <= owned:
        raise RuntimeError('SERVER_PORT_NOT_EXCLUSIVELY_OWNED')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    a = ap.parse_args()
    config = json.loads(a.config.read_text())
    required = {'command', 'env', 'url', 'out', 'gate_args', 'startup_seconds', 'run_seconds', 'metrics_path'}
    if set(config) != required:
        raise ValueError('CONFIG_FIELDS_MISMATCH')
    if not isinstance(config['command'], list) or not config['command']:
        raise ValueError('SERVER_COMMAND_REQUIRED')
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'launch.json').write_text(json.dumps(config, indent=2) + '\n')
    endpoint = urllib.parse.urlsplit(config['url'])
    if endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or endpoint.port is None:
        raise ValueError('EXPLICIT_LOOPBACK_HTTP_ENDPOINT_REQUIRED')
    if listening_inodes(endpoint.port):
        (out / 'run-receipt.json').write_text(json.dumps({'status':'failed', 'stage':'startup',
            'error':{'kind':'PortOwnershipError', 'detail':'SERVER_PORT_ALREADY_LISTENING'}}) + '\n')
        return 1
    executable = Path(config['command'][0]).resolve(strict=True)
    digest = hashlib.sha256()
    with executable.open('rb') as binary:
        for block in iter(lambda: binary.read(1048576), b''):
            digest.update(block)
    (out / 'executable.json').write_text(json.dumps({'path': str(executable), 'sha256': digest.hexdigest()}, indent=2) + '\n')
    gate = Path(__file__).resolve().parents[1] / 'quality_gate.py'
    with (out / 'server.log').open('w') as log:
        server = subprocess.Popen(config['command'], env={**os.environ, **config['env']}, stdout=log, stderr=subprocess.STDOUT)
        stage = 'startup'
        try:
            deadline = time.monotonic() + config['startup_seconds']
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'SERVER_EXIT_BEFORE_READY:{server.returncode}')
                try:
                    if owns_listener(server.pid, endpoint.port):
                        with urllib.request.urlopen(config['url'] + '/v1/models', timeout=2) as response:
                            models = json.load(response)
                        if not models.get('data'):
                            raise ValueError('MODEL_LIST_EMPTY')
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('SERVER_START_TIMEOUT')
                time.sleep(0.2)
            (out / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
            def metrics(name):
                if config['metrics_path'] is None:
                    return
                with urllib.request.urlopen(config['url'] + config['metrics_path'], timeout=10) as response:
                    data = json.load(response)
                (out / f'metrics-{name}.json').write_text(json.dumps(data, indent=2) + '\n')
            stage = 'metrics-before'
            metrics('before')
            stage = 'task-panel'
            result = subprocess.run([sys.executable, str(gate), '--server', config['url'], '--out', str(out), *config['gate_args']], timeout=config['run_seconds'])
            stage = 'metrics-after'
            metrics('after')
            (out / 'run-receipt.json').write_text(json.dumps({'status': 'complete' if result.returncode == 0 else 'failed', 'exit_code': result.returncode}) + '\n')
            return result.returncode
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            receipt = {'status': 'failed', 'stage': stage, 'server_exit_code': server.poll(),
                       'error': {'kind': type(error).__name__, 'detail': str(error)}}
            (out / 'run-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps(receipt), file=sys.stderr)
            return 1
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)


if __name__ == '__main__':
    sys.exit(main())
