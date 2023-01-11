def get_cv2_xy(height, width, x, y):
    """
    Args:
        height (float): environment height.
        width (float): environment width.
        x (float): x environment coordinate.
        y (float): y environment coordinate.

    Returns:
        y (int): x opencv coordinate.

        x (int): x opencv coordinate.
    """
    return int(height - y), int(x + (width / 2))
