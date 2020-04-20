import blimpy as bl
from turbo_seti.find_event import find_event, plot_event
import pylab as plt

def test_plot_hit():
    filename_dat = 'Voyager1.single_coarse.fine_res.dat'
    filename_fil = 'Voyager1.single_coarse.fine_res.h5'

    table = find_event.make_table(filename_dat)

    print(table)

    plt.subplot(3,1,1)
    plot_event.plot_hit(filename_fil, filename_dat, 0)
    plt.subplot(3,1,2)
    plot_event.plot_hit(filename_fil, filename_dat, 1)
    plt.subplot(3,1,3)
    plot_event.plot_hit(filename_fil, filename_dat, 2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_plot_hit()
