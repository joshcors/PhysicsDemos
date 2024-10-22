try:
    import cupy as xp
    from cupyx.scipy.interpolate import RegularGridInterpolator
except ModuleNotFoundError:
    import numpy as xp
    from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

class MandelbrotSet:
    def __init__(self, real_lower, real_upper, imag_lower, imag_upper, N_real, N_imag, min_res=(500, 500)):
        self.real_range = xp.array([real_lower, real_upper])
        self.imag_range = xp.array([imag_lower, imag_upper])

        self.real_view_range = self.real_range.copy()
        self.imag_view_range = self.imag_range.copy()

        self.min_res = xp.array(min_res)

        self.N_real = N_real
        self.N_imag = N_imag

        self.setup_grid()

        self.z = xp.zeros_like(self.c)
        self.z_mask = xp.ones(self.z.shape).astype(bool)

        self.increasing_res = False

        self.N_iterations = 0

        self.escape_time = xp.zeros((self.N_imag, self.N_real), dtype=xp.int32)

        red = list(xp.random.randint(0, 255, 10))
        green = list(xp.random.randint(0, 255, 10))
        blue = list(xp.random.randint(0, 255, 10))

        self.red   = xp.array([0, ] + red + [0, ])
        self.green = xp.array([0, ] + green + [0, ])
        self.blue  = xp.array([0, ] + blue + [0, ])

        indices = list(xp.random.randint(0, 255, 10))
        indices.sort()
        self.indices = xp.array([0, ] + indices + [255, ])

        self.red_interp = RegularGridInterpolator((self.indices, ), self.red)
        self.green_interp = RegularGridInterpolator((self.indices, ), self.green)
        self.blue_interp = RegularGridInterpolator((self.indices, ), self.blue)

    def color_interp(self, _in):
        r = self.red_interp(_in)
        g = self.green_interp(_in)
        b = self.blue_interp(_in)

        return xp.stack([r, g, b], axis=-1)

    def setup_grid(self):
        self.real = xp.linspace(*self.real_range, self.N_real)
        self.imag = xp.linspace(*self.imag_range, self.N_imag)

        self.d_real = xp.diff(self.real)[0]
        self.d_imag = xp.diff(self.imag)[0]
        
        real, imag = xp.meshgrid(self.real, self.imag)

        self.c = real + 1j * imag
        
    def iterate(self, N=1):
        if not self.increasing_res:
            for i in range(N):
                xp.putmask(self.z, self.z_mask, self.z * self.z + self.c)
                xp.putmask(self.z_mask, self.z_mask, xp.abs(self.z) < 2)
                xp.putmask(self.escape_time, self.z_mask, self.escape_time + 1)

            self.N_iterations += N

        else:
            increase_res_mask = xp.ones_like(self.z_mask)
            increase_res_mask[::2, ::2] = False

            for i in range(self.N_iterations):
                xp.putmask(self.z, increase_res_mask, self.z * self.z + self.c)
                xp.putmask(increase_res_mask, increase_res_mask, xp.abs(self.z) < 2)
                xp.putmask(self.escape_time, increase_res_mask, self.escape_time + 1)

            self.increasing_res = False

    def zoom(self, N=1):
        self.c = self.c[N:-N, N:-N]
        self.z = self.z[N:-N, N:-N]
        self.z_mask = self.z_mask[N:-N, N:-N]
        self.escape_time = self.escape_time[N:-N, N:-N]

        self.real = self.real[N:-N]
        self.imag = self.imag[N:-N]

        self.real_range += N * xp.array([self.d_real, -self.d_real])
        self.imag_range += N * xp.array([self.d_imag, -self.d_imag])

        self.N_real -= 2 * N
        self.N_imag -= 2 * N

        if (xp.array(self.escape_time.shape) < self.min_res).any():
            self.increase_resolution()

    def increase_resolution(self):
        self.z = xp.array(
            xp.insert(
                xp.insert(self.z.get(), xp.arange(-1, -self.N_imag, -1), 0, axis=0), xp.arange(-1, -self.N_real, -1), 0, axis=1
            )
        )

        self.z_mask = xp.array(
            xp.insert(
                xp.insert(self.z_mask.get(), xp.arange(-1, -self.N_imag, -1), True, axis=0), xp.arange(-1, -self.N_real, -1), True, axis=1
            )
        )

        self.escape_time = xp.array(
            xp.insert(
                xp.insert(self.escape_time.get(), xp.arange(-1, -self.N_imag, -1), 0, axis=0), xp.arange(-1, -self.N_real, -1), 0, axis=1
            )
        )

        self.N_real = 2 * self.N_real - 1
        self.N_imag = 2 * self.N_imag - 1

        self.real = xp.linspace(*self.real_range, self.N_real)
        self.imag = xp.linspace(*self.imag_range, self.N_imag)

        self.setup_grid()

        self.increasing_res = True
        self.iterate()

    def get_colors(self, z_grid):
        z_grid = (z_grid * 255 / z_grid.max())
        z_grid[z_grid <= 0] = 0.1
        z_grid[z_grid >= 255] = 254.9

        shape = z_grid.shape

        colors = self.color_interp((z_grid.flatten())).reshape(*shape, 3).astype(xp.uint8)

        return colors

    def get_image(self, resolution):
        interp = RegularGridInterpolator((xp.arange(self.N_imag), xp.arange(self.N_real)), self.escape_time, bounds_error=True)

        r_n = xp.linspace(0.1, self.N_real - 1.1, resolution[0])
        i_n = xp.linspace(0.1, self.N_imag - 1.1, resolution[1])

        image_grid = interp(tuple(xp.meshgrid(i_n, r_n, indexing="ij")))

        return self.get_colors(image_grid)
    
    def shape(self):
        return self.N_real, self.N_imag
    
    def get_hmap(self, resolution):
        interp = RegularGridInterpolator((xp.arange(self.N_imag), xp.arange(self.N_real)), self.escape_time, bounds_error=True)

        r_n = xp.linspace(0.1, self.N_real - 1.1, resolution[0])
        i_n = xp.linspace(0.1, self.N_imag - 1.1, resolution[1])

        image_grid = interp(tuple(xp.meshgrid(i_n, r_n, indexing="ij")))

        return image_grid

if __name__=="__main__":
    center = ( (-0.34853774148008254 -0.34831493420245574  ) / 2, (-0.6065922085831237 -0.606486596104741
) / 2)

    lower = (0.114, 0.627)
    upper = (0.122, 0.635)
    res = (2048, 2048)

    real = xp.linspace(lower[0], upper[0], res[0])
    imag = xp.linspace(lower[1], upper[1], res[1])

    ms = MandelbrotSet(lower[0], upper[0], lower[1], upper[1], *res, min_res=(801, 801))
    ms.iterate(3000)

    log_im = xp.log(xp.log(ms.escape_time.get()))
    log_im *= 255 / log_im.max()

    plt.pcolormesh(real, imag, log_im, cmap="gray")
    plt.show()

    quit()

    from scipy.ndimage import gaussian_filter

    log_im = gaussian_filter(log_im, 2)

    Image.fromarray(log_im.astype(xp.uint8), "L").save("test.png")

    quit()

    hmap = ms.get_hmap((1080, 1080)).get()

    hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min())

    import ipdb; ipdb.set_trace()

    hmap = xp.uint8(hmap_norm * 255)
    
    Image.fromarray(hmap, "L").save("hmap.png")

    quit()

    writer = imageio.get_writer("groovy_dark.mp4", fps=30)

    N = 500

    bar = Bar("Generating video...", max=N)

    ms.iterate(100)

    for i in range(N):
        ms.iterate(10)

        arr = ms.get_image((800, 800)).get()

        writer.append_data(arr)

        ms.zoom(15)
        bar.next()

    writer.close()
    bar.finish()

