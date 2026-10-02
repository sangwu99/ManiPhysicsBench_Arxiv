import pickle
import socket
import struct

def send(sock, obj):
    b = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack('!Q', len(b)) + b)

def recv(sock):
    buf = b''
    while len(buf) < 8:
        c = sock.recv(8 - len(buf))
        if not c:
            return None
        buf += c
    n = struct.unpack('!Q', buf)[0]
    body = b''
    while len(body) < n:
        c = sock.recv(min(1 << 20, n - len(body)))
        if not c:
            return None
        body += c
    return pickle.loads(body)

class PolicyClient:

    def __init__(self, host='127.0.0.1', port=7400):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _call(self, **kw):
        send(self.sock, kw)
        r = recv(self.sock)
        if r is None:
            raise RuntimeError('Policy server disconnected')
        if 'error' in r:
            raise RuntimeError(f"Policy server error: {r['error']}")
        return r

    def meta(self):
        return self._call(cmd='meta')

    def reset(self, instruction, episode=0):
        return self._call(cmd='reset', instruction=instruction, episode=episode)

    def infer(self, obs):
        return self._call(cmd='infer', obs=obs)

    def close(self):
        try:
            send(self.sock, dict(cmd='bye'))
        except OSError:
            pass
        self.sock.close()

def serve(port, handler, tag='policy'):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(4)
    print(f'[{tag}] listening on {port}', flush=True)
    while True:
        (conn, addr) = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f'[{tag}] client {addr}', flush=True)
        try:
            while True:
                req = recv(conn)
                if req is None or req.get('cmd') == 'bye':
                    break
                try:
                    send(conn, handler(req))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    send(conn, dict(error=f'{type(e).__name__}: {e}'))
        finally:
            conn.close()
            print(f'[{tag}] client gone', flush=True)
