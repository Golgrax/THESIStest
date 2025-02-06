# utils.py
def rescale_bbox(bbox, src_width, src_height, dest_width, dest_height):
    x1, y1, x2, y2 = bbox
    scale_x = dest_width / src_width
    scale_y = dest_height / src_height
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]