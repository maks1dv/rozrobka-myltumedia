import numpy as np
from PIL import Image

def to_light(code):
    c = np.asarray(code, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045,
                    c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)

def to_code(light):
    l = np.asarray(light, dtype=np.float64)
    s = np.where(l <= 0.0031308,
                 l * 12.92,
                 1.055 * (l ** (1 / 2.4)) - 0.055)
    return s * 255.0

def make_board(size=512, cell=8):
    yy, xx = np.indices((size, size))
    return (((yy // cell + xx // cell) % 2) * 255).astype(np.uint8)

def box_average(img, k):
    h, w = img.shape[:2]
    return img.reshape(h // k, k, w // k, k).mean(axis=(1, 3))

def rgb_to_ycbcr(img):
    x = img.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y  =  0.299    * r + 0.587    * g + 0.114    * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5      * b
    cr = 128 + 0.5      * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr

def ycbcr_to_rgb(y, cb, cr):
    cb_s = cb - 128.0
    cr_s = cr - 128.0
    r = y                       + 1.402    * cr_s
    g = y - 0.344136 * cb_s     - 0.714136 * cr_s
    b = y + 1.772    * cb_s
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

def rgb_to_hsv(img):
    x = img.astype(np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1); mn = x.min(-1); d = mx - mn

    h = np.zeros_like(mx)
    m = (d != 0) & (mx == r); h[m] = (60 * ((g - b)[m] / d[m])) % 360
    m = (d != 0) & (mx == g); h[m] = 60 * ((b - r)[m] / d[m]) + 120
    m = (d != 0) & (mx == b); h[m] = 60 * ((r - g)[m] / d[m]) + 240

    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return h, s, mx

def hsv_to_rgb(h, s, v):
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c

    r_p = np.zeros_like(h)
    g_p = np.zeros_like(h)
    b_p = np.zeros_like(h)

    hi = (h / 60.0) % 6
    
    c0 = (0 <= hi) & (hi < 1)
    c1 = (1 <= hi) & (hi < 2)
    c2 = (2 <= hi) & (hi < 3)
    c3 = (3 <= hi) & (hi < 4)
    c4 = (4 <= hi) & (hi < 5)
    c5 = (5 <= hi) & (hi <= 6)

    r_p[c0], g_p[c0], b_p[c0] = c[c0], x[c0], 0
    r_p[c1], g_p[c1], b_p[c1] = x[c1], c[c1], 0
    r_p[c2], g_p[c2], b_p[c2] = 0, c[c2], x[c2]
    r_p[c3], g_p[c3], b_p[c3] = 0, x[c3], c[c3]
    r_p[c4], g_p[c4], b_p[c4] = x[c4], 0, c[c4]
    r_p[c5], g_p[c5], b_p[c5] = c[c5], 0, x[c5]

    rgb = np.stack([r_p + m, g_p + m, b_p + m], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

def main():
    print("=== Завдання 1 ===")
    mid = (to_light(0) + to_light(255)) / 2.0
    print("to_code(mid):", round(float(to_code(mid))))
    print("Light 128:", round(float(to_light(128)) * 100, 1), "%")
    print("Light 64: ", round(float(to_light(64)) * 100, 1), "%")
    print("Light 192:", round(float(to_light(192)) * 100, 1), "%")

    print("\n=== Завдання 2 ===")
    board = make_board(512, 8)
    
    naive_arr = box_average(board.astype(np.float64), 16)
    naive_img = np.clip(np.round(naive_arr), 0, 255).astype(np.uint8)
    Image.fromarray(naive_img).save('board_naive.bmp')

    board_light = to_light(board)
    correct_light = box_average(board_light, 16)
    correct_arr = to_code(correct_light)
    correct_img = np.clip(np.round(correct_arr), 0, 255).astype(np.uint8)
    Image.fromarray(correct_img).save('board_correct.bmp')

    print(f"Наївний середняк код:   {naive_arr.mean():.1f}")
    print(f"Правильний середняк код: {correct_arr.mean():.1f}")
    print(f"Оригінал світло:         {board_light.mean() * 100:.1f}%")
    print(f"Наївно зменшене світло:  {to_light(naive_arr).mean() * 100:.1f}%")
    print(f"Правильно зменш. світло: {correct_light.mean() * 100:.1f}%")

    input_image = Image.open('photo.jpg').convert('RGB')
    img_rgb = np.array(input_image)

    print("\n=== Завдання 3 ===")
    y, cb, cr = rgb_to_ycbcr(img_rgb)
    restored_ycbcr = ycbcr_to_rgb(y, cb, cr)
    max_diff_ycbcr = np.abs(restored_ycbcr.astype(int) - img_rgb.astype(int)).max()
    print("Максимальна різниця RGB -> YCbCr -> RGB:", max_diff_ycbcr)

    gray_y = np.stack([y, y, y], axis=-1)
    gray_y = np.clip(np.round(gray_y), 0, 255).astype(np.uint8)
    Image.fromarray(gray_y).save('photo_y_channel.png')

    naive_bw = img_rgb.mean(axis=-1)
    gray_naive = np.stack([naive_bw, naive_bw, naive_bw], axis=-1)
    gray_naive = np.clip(np.round(gray_naive), 0, 255).astype(np.uint8)
    Image.fromarray(gray_naive).save('photo_naive_gray.png')

    print("\n=== Завдання 4 ===")
    h, s, v = rgb_to_hsv(img_rgb)
    restored_hsv = hsv_to_rgb(h, s, v)
    max_diff_hsv = np.abs(restored_hsv.astype(int) - img_rgb.astype(int)).max()
    print("Максимальна різниця RGB -> HSV -> RGB:", max_diff_hsv)

    exp1 = hsv_to_rgb(h, np.clip(s * 0.3, 0, 1), v)
    Image.fromarray(exp1).save('hsv_saturation_low.png')

    exp2 = hsv_to_rgb(h, np.clip(s * 2.0, 0, 1), v)
    Image.fromarray(exp2).save('hsv_saturation_high.png')

    exp3 = hsv_to_rgb((h + 60.0) % 360, s, v)
    Image.fromarray(exp3).save('hsv_hue_shifted.png')

    print("\nУсі файли успішно збережені в папці!")

if __name__ == "__main__":
    main()