from manim import *
import numpy as np

class ConvolutionWithArea(Scene):
    def construct(self):
        # Axes setup
        axes = Axes(
            x_range=[-5, 5],
            y_range=[0, 2],
            x_length=10,
            y_length=3,
            axis_config={"include_numbers": True}
        ).to_edge(UP)
        self.add(axes)

        # Define f(τ)
        def f_func(x):
            return np.where((x >= -1) & (x <= 1), 1, 0)

        f_graph = axes.plot(lambda x: f_func(x), color=BLUE, stroke_width=4)
        f_label = axes.get_graph_label(f_graph, label="f", x_val=-2)
        self.play(Create(f_graph), Write(f_label))

        # g(t - τ) with tracker
        t_tracker = ValueTracker(-2)

        def g_shifted(x):
            return np.exp(-(x - t_tracker.get_value())**2)

        g_graph = always_redraw(
            lambda: axes.plot(lambda x: g_shifted(x), color=PURPLE, stroke_width=4)
        )
        g_label = always_redraw(
            lambda: axes.get_graph_label(g_graph, label="g", x_val=2)
        )
        self.add(g_graph, g_label)

        # Overlap area
        def overlap(x):
            return f_func(x) * g_shifted(x)

        overlap_area = always_redraw(
            lambda: axes.get_area(
                axes.plot(lambda x: overlap(x), color=GREEN),
                x_range=[-5, 5],
                color="#00abab",
                opacity=1.0,
            )
        )
        self.add(overlap_area)

        # Animate convolution motion
        self.play(t_tracker.animate.set_value(2), run_time=6, rate_func=linear)
        self.wait()
