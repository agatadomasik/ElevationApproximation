import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_even_nodes(x_data, nodes_num):
    if nodes_num < 2:
        return [0]
    sec = (len(x_data) - 1) / (nodes_num - 1)
    return [int(sec * i) for i in range(nodes_num)]


def get_chebyshev_nodes(x_data, nodes_num):
    if nodes_num < 2:
        return [0]

    indices = []

    for k in range(nodes_num):
        index = int(((len(x_data) - 1) / 2) * (1 + np.cos(np.pi * (2 * k + 1) / (2 * nodes_num))))
        indices.append(index)

    return indices


def lagrange_basis(x_interp, x_points, i):
    phi = 1.0
    n = len(x_points)
    for j in range(n):
        if j != i:
            phi *= (x_interp - x_points[j]) / (x_points[i] - x_points[j])
    return phi


def lagrange_interpolation(x_interp, x_points, y_points):
    n = len(x_points)
    y_interpolated = 0.0
    for i in range(n):
        phi_i = lagrange_basis(x_interp, x_points, i)
        y_interpolated += y_points[i] * phi_i
    return y_interpolated


def spline_interpolation(x_data, y_data, x_interp):
    n = len(x_data) - 1
    h = [x_data[i + 1] - x_data[i] for i in range(n)]

    matrix_A = np.zeros((n + 1, n + 1))
    matrix_B = np.zeros(n + 1)

    matrix_A[0, 0] = 1
    matrix_A[n, n] = 1

    for i in range(1, n):
        matrix_A[i, i - 1] = h[i - 1]
        matrix_A[i, i] = 2 * (h[i - 1] + h[i])
        matrix_A[i, i + 1] = h[i]
        matrix_B[i] = 3 * ((y_data[i + 1] - y_data[i]) / h[i] - (y_data[i] - y_data[i - 1]) / h[i - 1])

    coeffs_c = np.linalg.solve(matrix_A, matrix_B)

    coeffs_a = y_data[:-1]
    coeffs_b = [(y_data[i + 1] - y_data[i]) / h[i] - (2 * coeffs_c[i] + coeffs_c[i + 1]) * h[i] / 3
                      for i in range(n)]
    coeffs_d = [(coeffs_c[i + 1] - coeffs_c[i]) / (3 * h[i]) for i in range(n)]

    interpolated_values = []
    for x in x_interp:
        for i in range(n):
            if x_data[i] <= x <= x_data[i + 1]:
                dx = x - x_data[i]
                interpolated_values.append(
                    coeffs_a[i] +
                    coeffs_b[i] * dx +
                    coeffs_c[i] * dx ** 2 +
                    coeffs_d[i] * dx ** 3
                )
                break
        else:
            interpolated_values.append(None)

    return interpolated_values


def plot_data(x, y, x_interp, y_interp, x_points, y_points, title, filename):
    # plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label="Dane oryginalne")
    plt.plot(x_interp, y_interp, 'r-', label="Funkcja interpolująca")
    plt.plot(x_points, y_points, 'ko', label="Punkty interpolacyjne")
    plt.title(title)
    plt.xlabel("dystans (m)")
    plt.ylabel("wysokosc (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/' + filename + '.png')
    plt.show()


def process_data(x_data, y_data, path):
    # interpolacja lagranga
    # równomierny rozkład węzłów
    indices_6 = get_even_nodes(x_data, 6)
    indices_10 = get_even_nodes(x_data, 10)
    indices_13 = get_even_nodes(x_data, 13)
    indices_20 = get_even_nodes(x_data, 20)
    indices_30 = get_even_nodes(x_data, 30)

    x_even_6 = [x_data[i] for i in indices_6]
    y_even_6 = [y_data[i] for i in indices_6]
    x_even_10 = [x_data[i] for i in indices_10]
    y_even_10 = [y_data[i] for i in indices_10]
    x_even_13 = [x_data[i] for i in indices_13]
    y_even_13 = [y_data[i] for i in indices_13]
    x_even_20 = [x_data[i] for i in indices_20]
    y_even_20 = [y_data[i] for i in indices_20]
    x_even_30 = [x_data[i] for i in indices_30]
    y_even_30 = [y_data[i] for i in indices_30]

    y_lagrange_6 = [lagrange_interpolation(x_data[i], x_even_6, y_even_6) for i in range(len(x_data))]
    y_lagrange_10 = [lagrange_interpolation(x_data[i], x_even_10, y_even_10) for i in range(len(x_data))]
    y_lagrange_13 = [lagrange_interpolation(x_data[i], x_even_13, y_even_13) for i in range(len(x_data))]
    y_lagrange_20 = [lagrange_interpolation(x_data[i], x_even_20, y_even_20) for i in range(len(x_data))]

    # wezly czebyszewa
    indices_cheb_6 = get_chebyshev_nodes(x_data, 6)
    indices_cheb_10 = get_chebyshev_nodes(x_data, 10)
    indices_cheb_13 = get_chebyshev_nodes(x_data, 13)
    indices_cheb_20 = get_chebyshev_nodes(x_data, 20)

    x_cheb_6 = [x_data[i] for i in indices_cheb_6]
    y_cheb_6 = [y_data[i] for i in indices_cheb_6]
    x_cheb_10 = [x_data[i] for i in indices_cheb_10]
    y_cheb_10 = [y_data[i] for i in indices_cheb_10]
    x_cheb_13 = [x_data[i] for i in indices_cheb_13]
    y_cheb_13 = [y_data[i] for i in indices_cheb_13]
    x_cheb_20 = [x_data[i] for i in indices_cheb_20]
    y_cheb_2O = [y_data[i] for i in indices_cheb_20]

    y_lagrange_cheb_6 = [lagrange_interpolation(x_data[i], x_cheb_6, y_cheb_6) for i in range(len(x_data))]
    y_lagrange_cheb_10 = [lagrange_interpolation(x_data[i], x_cheb_10, y_cheb_10) for i in range(len(x_data))]
    y_lagrange_cheb_13 = [lagrange_interpolation(x_data[i], x_cheb_13, y_cheb_13) for i in range(len(x_data))]
    y_lagrange_cheb_20 = [lagrange_interpolation(x_data[i], x_cheb_20, y_cheb_2O) for i in range(len(x_data))]

    # interpolacja funkcjami sklejanymi
    y_spline_10 = spline_interpolation(x_even_10, y_even_10, x_data)
    y_spline_13 = spline_interpolation(x_even_13, y_even_13, x_data)
    y_spline_20 = spline_interpolation(x_even_20, y_even_20, x_data)
    y_spline_30 = spline_interpolation(x_even_30, y_even_30, x_data)

    plot_data(x_data, y_data, x_data, y_lagrange_6, x_even_6, y_even_6, "Interpolacja Lagranga dla 6 węzłów o\n równomiernym rozkładzie - " + path,
        path + "_lag_6")
    plot_data(x_data, y_data, x_data, y_lagrange_10, x_even_10, y_even_10, "Interpolacja Lagranga dla 10 węzłów o\n równomiernym rozkładzie - " + path,
        path + "_lag_10")
    plot_data(x_data, y_data, x_data, y_lagrange_13, x_even_13, y_even_13, "Interpolacja Lagranga dla 13 węzłów o\n równomiernym rozkładzie - " + path,
        path + "_lag_13")
    plot_data(x_data, y_data, x_data, y_lagrange_20, x_even_20, y_even_20, "Interpolacja Lagranga dla 20 węzłów o\n równomiernym rozkładzie - " + path,
        path + "_lag_20")

    plot_data(x_data, y_data, x_data, y_lagrange_cheb_6, x_cheb_6, y_cheb_6, "Interpolacja Lagranga dla 6 węzłów\n Czebyszewa - " + path,
        path + "_lag_6_cheb")
    plot_data(x_data, y_data, x_data, y_lagrange_cheb_10, x_cheb_10, y_cheb_10, "Interpolacja Lagranga dla 10 węzłów\n Czebyszewa - " + path,
        path + "_lag_10_cheb")
    plot_data(x_data, y_data, x_data, y_lagrange_cheb_13, x_cheb_13, y_cheb_13, "Interpolacja Lagranga dla 13 węzłów\n Czebyszewa - " + path,
        path + "_lag_13_cheb")
    plot_data(x_data, y_data, x_data, y_lagrange_cheb_20, x_cheb_20, y_cheb_2O, "Interpolacja Lagranga dla 20 węzłów\n Czebyszewa - " + path,
        path + "_lag_20_cheb")

    plot_data(x_data, y_data, x_data, y_spline_10, x_even_10, y_even_10, "Interpolacja funkcjami sklejanymi 3 stopnia\n dla 10 węzłów - " + path,
        path + "_spl_10")
    plot_data(x_data, y_data, x_data, y_spline_13, x_even_13, y_even_13, "Interpolacja funkcjami sklejanymi 3 stopnia\n dla 13 węzłów - " + path,
        path + "_spl_13")
    plot_data(x_data, y_data, x_data, y_spline_20, x_even_20, y_even_20, "Interpolacja funkcjami sklejanymi 3 stopnia\n dla 20 węzłów - " + path,
        path + "_spl_20")
    plot_data(x_data, y_data, x_data, y_spline_30, x_even_30, y_even_30, "Interpolacja funkcjami sklejanymi 3 stopnia\n dla 30 węzłów - " + path,
        path + "_spl_30")


data1 = pd.read_csv("paths/MountEverest.csv")
x_data1 = data1.iloc[:, 0].tolist()
y_data1 = data1.iloc[:, 1].tolist()

data2 = pd.read_csv("paths/genoa_rapallo.txt", sep=" ")
x_data2 = data2.iloc[:, 0].tolist()
y_data2 = data2.iloc[:, 1].tolist()

data3 = pd.read_csv("paths/GlebiaChallengera.csv")
x_data3 = data3.iloc[:, 0].tolist()
y_data3 = data3.iloc[:, 1].tolist()

process_data(x_data1, y_data1, "Mount_Everest")
process_data(x_data2, y_data2, "Genoa_Rapallo")
# process_data(x_data3, y_data3, "Glebia_Challengera")
