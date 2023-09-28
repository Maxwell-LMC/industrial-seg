def generate_rect_from_point(x,y,side,type="screw"):
    if type == "screw":
        return (x-side//2, y-side//2, side, side)