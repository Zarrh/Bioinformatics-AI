from manim import *

class GradientDescent1D(Scene):
    def construct(self):
        def func(x):
            return x**2

        def grad(x):
            return 2 * x

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 16, 2],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True},
        )
        labels = axes.get_axis_labels("x", "f(x)")
        self.play(Create(axes), Write(labels))

        graph = axes.plot(func, color=BLUE)
        graph_label = axes.get_graph_label(graph, label="f(x) = x^2")
        self.play(Create(graph), Write(graph_label))

        x_tracker = ValueTracker(3.5)

        dot = always_redraw(lambda: Dot(
            axes.c2p(x_tracker.get_value(), func(x_tracker.get_value())), color=YELLOW)
        )
        dot_label = always_redraw(lambda: MathTex(
            f"x={x_tracker.get_value():.2f}"
        ).scale(0.7).next_to(dot, UP))

        self.play(FadeIn(dot), Write(dot_label))

        learning_rate = 0.3
        steps = 10

        for _ in range(steps):
            x_val = x_tracker.get_value()
            grad_val = grad(x_val)
            new_x = x_val - learning_rate * grad_val

            # Arrow tangent to the curve (aligned to gradient direction)
            dx = 0.5  # small horizontal length for visual arrow
            dy = grad_val * dx

            start_point = axes.c2p(x_val - dx / 2, func(x_val) - dy / 2)
            end_point = axes.c2p(x_val + dx / 2, func(x_val) + dy / 2)

            arrow = Arrow(start=end_point, end=start_point, color=PURPLE, buff=0, stroke_width=3)

            self.play(GrowArrow(arrow), run_time=0.4)
            self.play(x_tracker.animate.set_value(new_x), run_time=0.6)
            self.remove(arrow)
            self.wait(0.2)

        self.wait(1)
