from keras.preprocessing.image import *
import pickle
import cv2


# Modified directory generator
class DirectoryIterator(Iterator):
    def __init__(self, directory, image_data_generator, classes, samples, filenames, num_class,
                 target_size=(256, 256), color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format()

        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

        # first, count the number of samples and classes
        self.samples = 0

        self.num_class = num_class
        self.class_indices = dict(zip(classes, range(len(classes))))

        self.samples = samples
        self.filenames = filenames

        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_x = np.zeros((len(index_array),) + (160, 160, 3), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]

            # # These blocks were replaced
            # img = load_img(os.path.join(self.directory, fname),
            #                grayscale=grayscale,
            #                target_size=self.target_size)
            # x = img_to_array(img, data_format=self.data_format)
            x = cv2.imread(os.path.join(self.directory, fname))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)[0]
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        return self._get_batches_of_transformed_samples(index_array)


def get_train_gen(img_gen, batch_size=32, target_size=256):
    info = pickle.load(open("/shared/miha/ws/competitions/p18french/train_generator.pkl", "rb"))
    train_generator = DirectoryIterator('/workspace6/miha_misc2/train/', img_gen, classes=info["classes"],
                                        samples=info["samples"], filenames=info["filenames"], batch_size=batch_size,
                                        target_size=(target_size, target_size), num_class=info["num_class"])
    return train_generator


def get_val_gen(img_gen, batch_size=32, target_size=256):
    info = pickle.load(open("/shared/miha/ws/competitions/p18french/val_generator.pkl", "rb"))
    val_generator = DirectoryIterator('/workspace6/miha_misc2/val/', img_gen, classes=info["classes"],
                                      samples=info["samples"], filenames=info["filenames"], batch_size=batch_size,
                                      target_size=(target_size, target_size), num_class=info["num_class"])
    return val_generator


def get_custom_gen(pkl_file, img_gen, batch_size=32, target_size=256):
    info = pickle.load(open(pkl_file, "rb"))
    val_generator = DirectoryIterator('/workspace6/miha_misc2/train/', img_gen, classes=info["classes"],
                                      samples=info["samples"], filenames=info["filenames"], batch_size=batch_size,
                                      target_size=(target_size, target_size), num_class=info["num_class"])
    return val_generator


