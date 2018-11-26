import logging
import numpy as np
from tensorboard import SummaryWriter


def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def imageFromTensor(X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255./2.0), 0, 255).astype(np.uint8)
    n = np.int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]),dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])

    return buff
    #buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    #plt.imshow(title, buff)
    #cv2.waitKey(1)

class LogMetricsCallback(object):
    """Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but write TensorBoard event file
    for visualization. For more usage, please refer https://github.com/dmlc/tensorboard

    Parameters
    ----------
    logging_dir : str
        TensorBoard event file directory.
        After that, use `tensorboard --logdir=path/to/logs` to launch TensorBoard visualization.
    prefix : str
        Prefix for a metric name of `scalar` value.
        You might want to use this param to leverage TensorBoard plot feature,
        where TensorBoard plots different curves in one graph when they have same `name`.
        The follow example shows the usage(how to compare a train and eval metric in a same graph).

    Examples
    --------
    >>> # log train and eval metrics under different directories.
    >>> training_log = 'logs/train'
    >>> evaluation_log = 'logs/eval'
    >>> # in this case, each training and evaluation metric pairs has same name, you can add a prefix
    >>> # to make it separate.
    >>> batch_end_callbacks = [mx.tensorboard.LogMetricsCallback(training_log)]
    >>> eval_end_callbacks = [mx.tensorboard.LogMetricsCallback(evaluation_log)]
    >>> # run
    >>> model.fit(train,
    >>>     ...
    >>>     batch_end_callback = batch_end_callbacks,
    >>>     eval_end_callback  = eval_end_callbacks)
    >>> # Then use `tensorboard --logdir=logs/` to launch TensorBoard visualization.
    """
    def __init__(self, logging_dir, score_store=False, prefix=None):
        self.prefix = prefix
        self.step = 0
        self.score_store = score_store
        try:
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log training speed and metrics in TensorBoard."""
        self.step += 1
        if param.eval_metric is None:
            return
        name_value = param.eval_metric.get_name_value()
        if self.step % 20 == 0:
            for name, value in name_value:
                if self.prefix is not None:
                    name = '%s-%s' % (self.prefix, name)
                self.summary_writer.add_scalar(name, value,self.step)
        if self.step % 1000 == 0:
            im_ori = param.locals['data_batch'].label[0].asnumpy()
            im_rec = (param.locals['rec_img'])[0].asnumpy()
            im_ori = imageFromTensor(im_ori)
            im_rec = imageFromTensor(im_rec)
            self.summary_writer.add_image('im_ori',im_ori,self.step) 
            self.summary_writer.add_image('im_rec',im_rec,self.step)

            if self.score_store:
                facenet_scores = param.locals['facenet_scores']
                self.summary_writer.add_scalar('scores_mean', facenet_scores.mean(), self.step)
                self.summary_writer.add_histogram('facenet_scores', facenet_scores, self.step)

