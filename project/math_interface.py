import graph_builder
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from interface.streamlit_utils import render_function

import tinytorch
from tinytorch import MathTest, MathTestVariable

MyModule = None
tinytorch


def render_math_sandbox(use_scalar=False, use_tensor=False):
    st.write("## Sandbox for Math Functions")
    st.write("Visualization of the mathematical tests run on the underlying code.")

    if use_scalar:
        one, two, red = MathTestVariable._tests()
    else:
        one, two, red = MathTest._tests()
    f_type = st.selectbox("Function Type", ["One Arg", "Two Arg", "Reduce"])
    select = {"One Arg": one, "Two Arg": two, "Reduce": red}

    fn = st.selectbox("Function", select[f_type], format_func=lambda a: a[0])
    name, _, scalar = fn
    if f_type == "One Arg":
        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            if use_tensor:
                ys = [scalar(tinytorch.tensor([p]))[0] for p in xs]
            else:
                ys = [scalar(tinytorch.Scalar(p)).data for p in xs]
        else:
            ys = [scalar(p) for p in xs]
        scatter = go.Scatter(mode="lines", x=xs, y=ys)
        fig = go.Figure(scatter)
        st.write(fig)

        if use_scalar:
            st.write("Derivative f'(x)")
            if use_tensor:
                x_var = [tinytorch.tensor(x, requires_grad=True) for x in xs]
            else:
                x_var = [tinytorch.Scalar(x) for x in xs]
            for x in x_var:
                out = scalar(x)
                if use_tensor:
                    out.backward(tinytorch.tensor([1.0]))
                else:
                    out.backward()
            if use_tensor:
                scatter = go.Scatter(mode="lines", x=xs, y=[x.grad[0] for x in x_var])
            else:
                scatter = go.Scatter(
                    mode="lines", x=xs, y=[x.derivative for x in x_var]
                )
            fig = go.Figure(scatter)
            st.write(fig)
            G = graph_builder.GraphBuilder().run(out)
            G.graph["graph"] = {"rankdir": "LR"}
            st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

    if f_type == "Two Arg":

        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x, y)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            if use_tensor:
                zs = [
                    [
                        scalar(tinytorch.tensor([x]), tinytorch.tensor([y]))[0]
                        for x in xs
                    ]
                    for y in ys
                ]
            else:
                zs = [
                    [scalar(tinytorch.Scalar(x), tinytorch.Scalar(y)).data for x in xs]
                    for y in ys
                ]
        else:
            zs = [[scalar(x, y) for x in xs] for y in ys]

        scatter = go.Surface(x=xs, y=ys, z=zs)

        fig = go.Figure(scatter)
        st.write(fig)
        if use_scalar:
            a, b = [], []
            for x in xs:
                oa, ob = [], []

                if use_tensor:
                    for y in ys:
                        x1 = tinytorch.tensor([x])
                        y1 = tinytorch.tensor([y])
                        out = scalar(x1, y1)
                        out.backward(tinytorch.tensor([1]))
                        oa.append((x, y, x1.derivative[0]))
                        ob.append((x, y, y1.derivative[0]))
                else:
                    for y in ys:
                        x1 = tinytorch.Scalar(x)
                        y1 = tinytorch.Scalar(y)
                        out = scalar(x1, y1)
                        out.backward()
                        oa.append((x, y, x1.derivative))
                        ob.append((x, y, y1.derivative))
                a.append(oa)
                b.append(ob)
            st.write("Derivative f'_x(x, y)")

            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in a],
                y=[[c[1] for c in a2] for a2 in a],
                z=[[c[2] for c in a2] for a2 in a],
            )
            fig = go.Figure(scatter)
            st.write(fig)
            st.write("Derivative f'_y(x, y)")
            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in b],
                y=[[c[1] for c in a2] for a2 in b],
                z=[[c[2] for c in a2] for a2 in b],
            )
            fig = go.Figure(scatter)
            st.write(fig)
    if f_type == "Reduce":
        st.write("### " + name)
        render_function(scalar)
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]

        if use_tensor:
            scatter = go.Surface(
                x=xs,
                y=ys,
                z=[[scalar(tinytorch.tensor([x, y]))[0] for x in xs] for y in ys],
            )
        else:
            scatter = go.Surface(
                x=xs, y=ys, z=[[scalar([x, y]) for x in xs] for y in ys]
            )
        fig = go.Figure(scatter)
        st.write(fig)
