import tornado.ioloop
import tornado.web
from handler.data_handler import DataHandler
from handler.train_handler import TrainHandler
from handler.predict_handler import PredictHandler
from handler.split_handler import SplitHandler
from handler.images_handler import ImagesHandler
from handler.confusion_handler import ConfusionHandler
from handler.threshold_handler import ThresholdHandler


def make_app():
    return tornado.web.Application([
        (r"/demo/data", DataHandler),
        (r"/demo/split", SplitHandler),
        (r"/demo/tain", TrainHandler),
        (r"/demo/predict", PredictHandler),
        (r"/demo/confusion", ConfusionHandler),
        (r"/demo/threshold", ThresholdHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8881)
    tornado.ioloop.IOLoop.current().start()

