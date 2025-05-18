from manim import *
from math import *

class NeuralNetwork(Scene):
  def construct(self):
    layers = [3, 5, 4, 2]
    layer_distance = 3.5
    nodes_distance = 1.5
    node_radius = 0.5

    all_nodes = []

    self.camera.frame_center = [(len(layers)-1) * layer_distance / 2, 0, 0]

    for layer_idx, num_nodes in enumerate(layers):
      layer_nodes = []
      for node_idx in range(num_nodes):
        y_offset = (num_nodes - 1) * nodes_distance / 2
        y_pos = y_offset - node_idx * nodes_distance
        position = RIGHT * layer_idx * layer_distance + UP * y_pos

        border = Annulus(
          inner_radius=node_radius * 0.85,
          outer_radius=node_radius * 1.2,
          color=BLUE,
          stroke_width=0,
          z_index=20,
        ).move_to(position)
        border.set_fill(
          color=BLUE,
          opacity=1
        )
        border.set_color(WHITE) 

        border.set_color([BLUE, PURPLE])

        node = Circle(radius=node_radius, fill_opacity=0, color=WHITE, stroke_width=0).move_to(position)
        self.add(border, node)
        layer_nodes.append(node)
      all_nodes.append(layer_nodes)

    for i in range(len(all_nodes) - 1):
      for node_a in all_nodes[i]:
        for node_b in all_nodes[i + 1]:
          line = Line(
            node_a.get_right(), 
            node_b.get_left(), 
            stroke_width=5.0, 
            fill_opacity=0.8, 
            color="#a3a2a2"
          )
          self.add(line)


class Neuron(Scene):
  def construct(self):
    node_radius = 1.75

    # self.camera.frame_center = [(len(layers)-1) * layer_distance / 2, 0, 0]

    position = [0, 0, 0]

    border = Annulus(
      inner_radius=node_radius * 0.85,
      outer_radius=node_radius * 1.2,
      color=BLUE,
      stroke_width=0,
      z_index=20,
    ).move_to(position)
    border.set_fill(
      color=BLUE,
      opacity=1
    )
    border.set_color(WHITE) 

    border.set_color([BLUE, PURPLE])

    node = Circle(radius=node_radius, fill_opacity=0, color=WHITE, stroke_width=0).move_to(position)
    self.add(border, node)

    line_0 = Line(
      node.get_left(), 
      node.get_left() - np.array([4.5, 0, 0]), 
      stroke_width=10.0, 
      color="#d4d4d4"
    )
    self.add(line_0)

    label_0 = MathTex("x_1", font_size=84).set_color("#ffffff").move_to(node.get_left() + np.array([-2.25, 0.5, 0]))
    self.add(label_0)


    line_1 = Line(
      np.append((np.array([[sqrt(2)/2, -sqrt(2)/2], [sqrt(2)/2, sqrt(2)/2]]) @ node.get_left()[:2]), [0]), 
      np.append(4.5*(np.array([[sqrt(2)/2, -sqrt(2)/2], [sqrt(2)/2, sqrt(2)/2]]) @ node.get_left()[:2]), [0]), 
      stroke_width=10.0, 
      color="#d4d4d4"
    )
    self.add(line_1)

    label_1 = MathTex("x_0", font_size=84).set_color("#ffffff").move_to(
      np.append(2.5*(np.array(
        [[sqrt(2)/2, sqrt(2)/2], [-sqrt(2)/2, sqrt(2)/2]]
      ) @ node.get_left()[:2]) + np.array([0.5, 0.5]), [0])
    )
    self.add(label_1)


    line_2 = Line(
      np.append((np.array([[sqrt(2)/2, sqrt(2)/2], [-sqrt(2)/2, sqrt(2)/2]]) @ node.get_left()[:2]), [0]), 
      np.append(4.5*(np.array([[sqrt(2)/2, sqrt(2)/2], [-sqrt(2)/2, sqrt(2)/2]]) @ node.get_left()[:2]), [0]), 
      stroke_width=10.0, 
      color="#d4d4d4"
    )
    self.add(line_2)

    label_2 = MathTex("x_i", font_size=84).set_color("#ffffff").move_to(
      np.append(2.5*(np.array(
        [[sqrt(2)/2, -sqrt(2)/2], [sqrt(2)/2, sqrt(2)/2]]
      ) @ node.get_left()[:2]) + np.array([0.5, -0.5]), [0])
    )
    self.add(label_2)


    line_3 = Line(
      node.get_right(), 
      node.get_right() + np.array([4.5, 0, 0]), 
      stroke_width=10.0, 
      color="#d4d4d4"
    )
    self.add(line_3)

    label_3 = MathTex("f(z)", font_size=84).set_color("#ffffff").move_to(
      node.get_right() + np.array([2.5, 0.7, 0])
    )
    self.add(label_3)


    center_label = MathTex("z", font_size=118).set_color("#ffffff").move_to(
      [0, 0, 0]
    )
    self.add(center_label)



class DigitsNN(Scene):
  def construct(self):
    size = (784, 40, 32, 16, 10)
    layers = [32, 40, 32, 16, 10]
    layer_distance = 2.0
    nodes_distance = 0.35
    node_radius = 0.15

    all_nodes = []

    self.camera.frame_center = [(len(layers)-1) * layer_distance / 2, 0, 0]

    for layer_idx, num_nodes in enumerate(layers):
      layer_nodes = []
      for node_idx in range(num_nodes):
        y_offset = (num_nodes - 1) * nodes_distance / 2
        y_pos = y_offset - node_idx * nodes_distance
        position = RIGHT * layer_idx * layer_distance + UP * y_pos

        if layer_idx == 0 and node_idx == num_nodes // 2:
          dot_spacing = 0.12
          for i in range(3):
            dot = Dot(point=position + RIGHT * dot_spacing * (i - 1), radius=0.02, color=WHITE)
            self.add(dot)
          continue

        border = Annulus(
          inner_radius=node_radius * 0.85,
          outer_radius=node_radius * 1.2,
          color=BLUE,
          stroke_width=0,
          z_index=20,
        ).move_to(position)
        border.set_fill(
          color=BLUE,
          opacity=1
        )
        border.set_color(WHITE) 

        border.set_color([BLUE, PURPLE])

        node = Circle(radius=node_radius, fill_opacity=0, color=WHITE, stroke_width=0).move_to(position)
        self.add(border, node)
        layer_nodes.append(node)
      all_nodes.append(layer_nodes)

    first_layer_nodes = all_nodes[0]
    top_node = first_layer_nodes[0]
    bottom_node = first_layer_nodes[-1]
    brace = BraceBetweenPoints(bottom_node.get_left(), top_node.get_left(), direction=LEFT)
    brace_text = brace.get_text("784", buff=0.15).scale(1.2)
    self.add(brace, brace_text)

    for i in range(len(all_nodes) - 1):
      for node_a in all_nodes[i]:
        for node_b in all_nodes[i + 1]:
          line = Line(
            node_a.get_right(), 
            node_b.get_left(), 
            stroke_width=1.5, 
            fill_opacity=0.5, 
            color="#a3a2a2"
          )
          self.add(line)



class Sigmoid(Scene):
  def construct(self):
    axes = Axes(
      x_range=[-10, 10, 2],
      y_range=[0, 1.1, 0.2],
      x_length=12,
      y_length=12*2/5,
      axis_config={"include_tip": True},
    )

    labels = axes.get_axis_labels(x_label="z", y_label="y")

    sigmoid_curve = axes.plot(
      lambda x: 1 / (1 + np.exp(-x)),
      color="#00abab",
      stroke_width=10,
    )

    self.add(axes, labels, sigmoid_curve)


class ReLU(Scene):
  def construct(self):
    axes = Axes(
      x_range=[-5, 5, 1],
      y_range=[-1, 5, 1],
      x_length=12,
      y_length=12*3/5,
      axis_config={"include_tip": True},
    )

    labels = axes.get_axis_labels(x_label="z", y_label="y")

    curve = axes.plot(
      lambda x: x if x > 0 else 0,
      color="#00abab",
      stroke_width=10,
    )

    self.add(axes, labels, curve)
