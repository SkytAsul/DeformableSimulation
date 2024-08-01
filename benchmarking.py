import time
from datetime import datetime
import bisect
import csv
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

class Plotter:
    def __init__(self, title = "Real-time plot"):
        self.title = title
        self.start()

    def start(self):
        self._csv_data = []
        self._plot_data = []
        self._iter_time = []
        self._labels = []
        self._iter_i = 0
        self._iter_j = 0
        self._animation = None
    
    def new_iteration(self):
        self._iter_data = [] if self._iter_i == 0 else [0] * len(self._labels)
        self._iter_time.append(datetime.now())
        self._iter_j = 0

    def plot(self, value, label):
        if self._iter_i == 0:
            self._iter_data.append(value)
            self._plot_data.append([value])
            self._labels.append(label)
        else:
            self._iter_data[self._iter_j] = value
            self._plot_data[self._iter_j].append(value)
        self._iter_j += 1

    def end_iteration(self):
        if self._iter_j != 0:
            self._csv_data.append(self._iter_data)
            self._iter_i += 1
    
    def stop(self):
        self.end_iteration()
        if self._animation != None:
            self._animation.pause()

    def get_data(self, from_date: datetime = None, from_index: int | None = None, only_columns: list[str] | None = None) -> tuple[dict[str, list[float]], datetime]:
        if from_date is not None and from_index is not None:
            raise ValueError("Cannot have both from_date and from_index")

        data = self._plot_data
        time = self._iter_time[0]
        if from_date is not None:
            from_index = bisect.bisect(self._iter_time, from_date)
        
        if from_index is not None:
            data = [col[from_index:] for col in data]
            time = self._iter_time[from_index]
        
        labelled_data = {}
        for i, label in enumerate(self._labels):
            if not only_columns or label in only_columns:
                labelled_data[label] = data[i]
        return labelled_data, time

    def export_csv(self, path, include_iter = False, include_time = False):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            headers = []
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

    def graph_viz(self, max_points = -1, use_time = False, y_axis = "Value"):
        """
        Displays a real-time line chart.

        This method will only returns when the user hits closes the visualization window,
        even if the 'stop' method is called beforehand.
        
        WARNING: This method must be called in the main thread.
        """
        figure, ax = pyplot.subplots()
        figure.canvas.manager.set_window_title(self.title)
        ax.set_xlabel('Time' if use_time else 'Iteration')
        ax.set_ylabel(y_axis)

        lines = []

        def update(_):
            changed_lines = False

            if use_time:
                x_data = [dt for dt in self._iter_time[0:self._iter_i]]
            else:
                x_data = range(self._iter_i)

            for i in range(0, len(self._plot_data)):
                if i >= len(lines):
                    line, = pyplot.plot([datetime.now() if use_time else 0], [1], label = self._labels[i])
                    lines.append(line)
                    changed_lines = True
                
                lines[i].set_data(x_data, self._plot_data[i][0:self._iter_i])
                
            x_shown = x_data[-max_points:]
            if len(x_shown) > 1:
                ax.set_xlim(left = x_shown[0], right = x_shown[-1])
            ax.relim()
            ax.autoscale_view()
            if changed_lines:
                figure.legend()
            
            return lines

        self._animation = FuncAnimation(figure, update, interval=300, cache_frame_data=False, blit=False)
        # no blit because axes change, legend...
        pyplot.show()
        self._animation = None

class Benchmarker(Plotter):
    def __init__(self, title = "Real-time benchmarking"):
        super().__init__(title)

    def new_iteration(self):
        super().new_iteration()
        self.begin_mark()

    def mark(self, label):
        duration = time.perf_counter() - self._time
        super().plot(duration, label)
        self.begin_mark()

    def begin_mark(self):
        self._time = time.perf_counter()
    
    def graph_viz(self, max_points = -1, use_time = False, y_axis = "Time taken (s)"):
        return super().graph_viz(max_points, use_time, y_axis)

# demo
if __name__ == "__main__":
    from threading import Thread
    import random
    b = Benchmarker()

    def d():
        for i in range(6):
            b.new_iteration()
            time.sleep(random.uniform(0.4, 0.6))
            b.mark("slept1")
            time.sleep(random.uniform(0.1, 0.3))
            b.mark("slept2")
            time.sleep(random.uniform(0.6, 0.8))
            b.mark("slept3")
            time.sleep(3 if i == 2 else 0.5)
            b.mark("slept4")
            b.end_iteration()
            print("Finished iter")
        b.stop()
        b.export_csv("test.csv", include_time=True)

    t = Thread(target=d)
    t.start()
    b.graph_viz(max_points=4, use_time=True)
