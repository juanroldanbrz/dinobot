from model.rectangle import Rectangle


def assert_rectangle_shape(img_np, rectangle: Rectangle, message: str):
    img_shape = img_np.shape

    assert img_shape[0] == rectangle.height(), message
    assert img_shape[1] == rectangle.width(), message


def assert_gray_img(img_np):
    assert len(img_np.shape) == 2, 'Image should be gray'

