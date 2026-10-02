import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

def start(spec, base, log):
    command = spec['command']
    if not isinstance(command, list) or not command or any((not isinstance(x, str) for x in command)):
        raise ValueError('Commands must be nonempty argument lists')
    environment = os.environ.copy()
    environment.update(spec.get('env', {}))
    cwd = (base / spec.get('cwd', '.')).resolve()
    return subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

def ready(process, spec):
    (host, port) = (spec['host'], int(spec['port']))
    deadline = time.monotonic() + float(spec.get('timeout_s', 600))
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f'Server exited before readiness with code {code}')
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except (ConnectionRefusedError, TimeoutError):
            time.sleep(0.2)
    raise TimeoutError('Server readiness deadline exceeded')

def run(config_path, output):
    (config_path, output) = (Path(config_path).resolve(), Path(output).resolve())
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    status = dict(state='running', jobs=[])
    status_path = output / 'status.json'
    server = None
    active = None
    server_log = None
    status_path.write_text(json.dumps(status, indent=2))
    try:
        if 'server' in config:
            server_spec = config['server']
            address = (server_spec['ready']['host'], int(server_spec['ready']['port']))
            with socket.socket() as probe:
                if probe.connect_ex(address) == 0:
                    raise RuntimeError('Server port is already occupied')
            server_log = (output / 'server.log').open('w')
            server = start(server_spec, config_path.parent, server_log)
            ready(server, server_spec['ready'])
        if not config['jobs']:
            raise ValueError('At least one rollout job is required')
        for (i, job) in enumerate(config['jobs']):
            name = job.get('name', str(i))
            log_name = f'job_{i:03d}.log'
            record = dict(name=name, log=log_name, state='running')
            status['jobs'].append(record)
            status_path.write_text(json.dumps(status, indent=2))
            with (output / log_name).open('w') as log:
                active = start(job, config_path.parent, log)
                while active.poll() is None:
                    if server is not None and server.poll() is not None:
                        raise RuntimeError(f'Server exited during job {name} with code {server.returncode}')
                    time.sleep(0.1)
                code = active.returncode
                record.update(returncode=code, state='completed' if code == 0 else 'failed')
                stop(active)
                active = None
                if code != 0:
                    raise subprocess.CalledProcessError(code, job['command'])
            if server is not None and server.poll() is not None:
                raise RuntimeError(f'Server exited during job {name} with code {server.returncode}')
        status['state'] = 'completed'
    except BaseException as error:
        status.update(state='failed', error=str(error))
        if status['jobs'] and status['jobs'][-1]['state'] == 'running':
            status['jobs'][-1]['state'] = 'failed'
        raise
    finally:
        stop(active)
        stop(server)
        if server_log is not None:
            server_log.close()
        status_path.write_text(json.dumps(status, indent=2))
    return status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.config, args.output)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
