import tornado.ioloop
import tornado.web
import os

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("base.html")

def make_app():
    return tornado.web.Application(
    	[
        (r"/", MainHandler),
    	],
    	templete_path=os.path.join(
    		os.path.dirname(__file__), "templates"
    	),
    	debug=True
    )

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()