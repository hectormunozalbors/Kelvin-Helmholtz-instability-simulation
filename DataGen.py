import numpy as np

def generate_dataset(
    nx=2000,
    L=1.0,
    np_interface=256000,
    radii=None,
):
    if radii is None:
        radii = np.concatenate([
            np.arange(0.00225, 0.051, 0.001),
            np.arange(0.075, 0.476, 0.025)
        ])

    ny = nx
    h = L / (nx - 1)
    x0, y0 = 0.5, 0.5

    datafra = []
    datacur = []

    dtheta = 2 * np.pi / (np_interface - 1)

    for ro in radii:
        #print(f"radius = {ro}")

        theta = np.arange(np_interface) * dtheta
        xp = x0 + ro * np.cos(theta)
        yp = y0 + ro * np.sin(theta)

        xi = np.zeros((nx - 1, ny - 1))

        for i in range(np_interface - 1):
            x1, y1 = xp[i], yp[i]
            x2, y2 = xp[i + 1], yp[i + 1]

            ip1 = int(np.floor(x1 * (nx - 1)))
            ip2 = int(np.floor(x2 * (nx - 1)))
            jp1 = int(np.floor(y1 * (ny - 1)))
            jp2 = int(np.floor(y2 * (ny - 1)))

            xs = [x1, x2, x2, x2]
            ys = [y1, y2, y2, y2]

            if ip1 != ip2:
                xv = (max(ip1, ip2)) * h
                yv = y1 + (y2 - y1) * (xv - x1) / (x2 - x1)
            else:
                xv = yv = None

            if jp1 != jp2:
                yh = (max(jp1, jp2)) * h
                xh = x1 + (x2 - x1) * (yh - y1) / (y2 - y1)
            else:
                xh = yh = None

            if ip1 != ip2 and jp1 == jp2:
                xs[1], ys[1] = xv, yv
            elif jp1 != jp2 and ip1 == ip2:
                xs[1], ys[1] = xh, yh
            elif ip1 != ip2 and jp1 != jp2:
                xs[1], ys[1] = xv, yv
                xs[2], ys[2] = xh, yh
                if (ip2 - ip1) * (xs[1] - xs[2]) < 0:
                    xs[1], xs[2] = xs[2], xs[1]
                    ys[1], ys[2] = ys[2], ys[1]

            for j in range(3):
                xm = 0.5 * (xs[j] + xs[j + 1])
                ym = 0.5 * (ys[j] + ys[j + 1])

                ip = int(np.floor(xm * (nx - 1)))
                jp = int(np.floor(ym * (ny - 1)))

                dx = -(xs[j + 1] - xs[j])
                xi[ip, jp] += (ym - jp * h) * dx / h**2
                xi[ip, :jp] += dx / h

        cellloc = []
        last = (-1, -1)
        for i in range(np_interface - 1):
            ip = int(np.floor(xp[i] * (nx - 1)))
            jp = int(np.floor(yp[i] * (ny - 1)))
            if (ip, jp) != last:
                cellloc.append((ip, jp))
                last = (ip, jp)

        if cellloc[0] == cellloc[-1]:
            cellloc = cellloc[:-1]

        for ip, jp in cellloc:
            cur = h / ro
            fra = xi[ip-1:ip+2, jp-1:jp+2].flatten()

            datacur.append(cur)
            datafra.append(fra)

    datacur = np.array(datacur)
    datafra = np.array(datafra)

    datacur = np.concatenate([datacur, -datacur])
    datafra = np.concatenate([datafra, 1.0 - datafra])

    return datafra, datacur

print(generate_dataset())
