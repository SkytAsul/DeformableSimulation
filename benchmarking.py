import time, datetime
import csv
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

class Benchmarker:
    def __init__(self):
        self.start()

    def start(self):
        self._csv_data = []
        self._plot_data = []
        self._iter_time = []
        self._labels = []
        self._iter_i = 0
    
    def new_iteration(self):
        self._time = time.perf_counter()
        self._iter_data = [] if self._iter_i == 0 else [0] * len(self._labels)
        self._iter_time.append(datetime.datetime.now())
        self._iter_j = 0

    def mark(self, label):
        duration = time.perf_counter() - self._time
        if self._iter_i == 0:
            self._iter_data.append(duration)
            self._plot_data.append([duration])
            self._labels.append(label)
        else:
            self._iter_data[self._iter_j] = duration
            self._plot_data[self._iter_j].append(duration)
        self._iter_j += 1
        
        self._time = time.perf_counter()

    def end_iteration(self):
        if self._iter_j != 0:
            self._csv_data.append(self._iter_data)
            self._iter_i += 1
    
    def stop(self):
        self.end_iteration()
        self._animation.pause()

    def export_csv(self, path, include_iter = False, include_time = False):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            headers = []
            data = self._csv_data.copy()
            if include_iter:
                headers.append("Iteration")
            if include_time:
                headers.append("Time")
            headers += self._labels
            writer.writerow(headers)

            for i in range(self._iter_i-1):
                row = []
                if include_iter:
                    row.append(i)
                if include_time:
                    row.append(self._iter_time[i])
                row += self._csv_data[i]
                writer.writerow(row)

    def graph_viz(self, max_points = -1, use_time = False):
        '''
        DO NOT CALL OUTSIDE THE MAIN THREAD
        '''
        figure, ax = pyplot.subplots()
        figure.canvas.manager.set_window_title('Real-time benchmarking')
        ax.set_xlabel('Time' if use_time else 'Iteration')
        ax.set_ylabel('Time taken (s)')

        lines = []

        def update(frame):
            changed_lines = False

            if use_time:
                x_data = [dt for dt in self._iter_time[0:self._iter_i]]
            else:
                x_data = range(self._iter_i)

            for i in range(0, len(self._plot_data)): # from 1 because there are the x values
                if i >= len(lines):
                    line, = pyplot.plot([datetime.datetime.now() if use_time else 0], [1], label = self._labels[i])
                    lines.append(line)
                    changed_lines = True
                
                lines[i-1].set_data(x_data, self._plot_data[i][0:self._iter_i])
                
            x_shown = x_data[-max_points:]
            if x_shown != []:
                ax.set_xlim(left = x_shown[0], right = x_shown[-1])
            ax.relim()
            ax.autoscale_view()
            if changed_lines:
                figure.legend()

        self._animation = FuncAnimation(figure, update, interval=300, cache_frame_data=False)
        pyplot.show()


# demo
from threading import Thread
import random
if __name__ == "__main__":
    b = Benchmarker()

    def d():
        for i in range(6):
            b.new_iteration()
            time.sleep(random.uniform(0.4, 0.6))
            b.mark("slept")
            time.sleep(random.uniform(0.1, 0.3))
            b.mark("slept2")
            time.sleep(random.uniform(0.6, 0.8))
            b.mark("slept3")
            b.end_iteration()
            print("Finished iter")
        b.stop()
        b.export_csv("test.csv", include_time=True)

    t = Thread(target=d)
    t.start()
    b.graph_viz(max_points=4, use_time=True)
