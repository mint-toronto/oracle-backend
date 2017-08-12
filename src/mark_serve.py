##
## Sample usage:
##
## submit POST:
##
## curl -H "Content-Type: application/json" -X POST -d '{"tt1": ["50", "0.25"], "qa1": ["85", "0.05"]}' http://localhost:8080
##
## @author Gideon Providence
##
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import simplejson

from model.markmodel import MarkModel

MODEL = MarkModel()

class Serve(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        self._set_headers()
        content_len = int(self.headers['Content-Length'])
        post_body   = self.rfile.read(content_len)
        data        = simplejson.loads(post_body)
        self.wfile.write(bytes("<html><body><h1>RECEIVED!</h1><br>", "utf-8"))
        self.wfile.write(bytes(repr(data), "utf-8"))
        self.wfile.write(bytes("<br></body></html>", "utf-8"))

class ThreadedServe(ThreadingMixIn, HTTPServer):
    pass
        
def run(server_class=HTTPServer, handler_class=Serve, port=8080):
    server = ThreadedServe(('localhost', 8080), Serve)
    print('Starting httpd...')
    server.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
