import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

G = 1.0


def n_body_derivatives(_t: float, y: np.ndarray, masses: np.ndarray) -> np.ndarray:
    positions = y[:9].reshape(3, 3)
    velocities = y[9:].reshape(3, 3)

    accelerations = np.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            distance = np.linalg.norm(diff) + 1e-9
            accelerations[i] += G * masses[j] * diff / distance**3

    return np.concatenate([velocities.ravel(), accelerations.ravel()])


def simulate_three_body(
    masses: np.ndarray,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    t_max: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    y0 = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])
    t_eval = np.linspace(0, t_max, steps)
    solution = solve_ivp(
        n_body_derivatives,
        (0, t_max),
        y0,
        args=(masses,),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-9,
    )
    positions = solution.y[:9].T.reshape(-1, 3, 3)
    return solution.t, positions


def make_trace(x: np.ndarray, y: np.ndarray, z: np.ndarray, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line={"color": color, "width": 3},
    )


st.set_page_config(page_title="3-body chaos explorer", layout="wide")
st.title("Explorateur du problème à trois corps")
st.caption("Visualisation d'une dynamique chaotique (attracteurs étranges possibles selon les conditions initiales).")

with st.sidebar:
    st.header("Paramètres")
    t_max = st.slider("Durée de simulation", 10.0, 200.0, 80.0, 5.0)
    steps = st.slider("Nombre de pas", 500, 10000, 3000, 250)

    st.subheader("Masses")
    m1 = st.number_input("Masse corps 1", min_value=0.01, value=1.0, step=0.05)
    m2 = st.number_input("Masse corps 2", min_value=0.01, value=1.0, step=0.05)
    m3 = st.number_input("Masse corps 3", min_value=0.01, value=1.0, step=0.05)

    st.subheader("Conditions initiales")
    preset = st.selectbox(
        "Preset",
        ["Figure-8 (chaotique doux)", "Triangle perturbé", "Libre"],
    )

if preset == "Figure-8 (chaotique doux)":
    initial_positions = np.array(
        [[-0.97000436, 0.24308753, 0.0], [0.97000436, -0.24308753, 0.0], [0.0, 0.0, 0.0]]
    )
    initial_velocities = np.array(
        [[0.4662036850, 0.4323657300, 0.0], [0.4662036850, 0.4323657300, 0.0], [-0.93240737, -0.86473146, 0.0]]
    )
elif preset == "Triangle perturbé":
    initial_positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.4, 0.0]])
    initial_velocities = np.array([[0.22, 0.35, 0.03], [0.18, -0.35, -0.02], [-0.38, 0.0, 0.01]])
else:
    cols = st.sidebar.columns(3)
    labels = ["x", "y", "z"]
    initial_positions = np.zeros((3, 3))
    initial_velocities = np.zeros((3, 3))
    for body in range(3):
        st.sidebar.markdown(f"**Corps {body+1}**")
        for axis, label in enumerate(labels):
            initial_positions[body, axis] = cols[axis].number_input(
                f"r{body+1}{label}", value=float((body - 1) * (axis == 0)), key=f"r{body}{label}"
            )
        for axis, label in enumerate(labels):
            initial_velocities[body, axis] = cols[axis].number_input(
                f"v{body+1}{label}", value=0.1 * (body - 1) * (axis == 1), key=f"v{body}{label}"
            )

masses = np.array([m1, m2, m3], dtype=float)

time_points, positions = simulate_three_body(masses, initial_positions, initial_velocities, t_max=t_max, steps=steps)

colors = ["#3b82f6", "#ef4444", "#10b981"]
names = ["Corps 1", "Corps 2", "Corps 3"]

fig3d = go.Figure()
for idx in range(3):
    fig3d.add_trace(make_trace(positions[:, idx, 0], positions[:, idx, 1], positions[:, idx, 2], names[idx], colors[idx]))
    fig3d.add_trace(
        go.Scatter3d(
            x=[positions[-1, idx, 0]],
            y=[positions[-1, idx, 1]],
            z=[positions[-1, idx, 2]],
            mode="markers",
            marker={"size": 5, "color": colors[idx]},
            showlegend=False,
        )
    )

fig3d.update_layout(
    height=700,
    scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
    margin={"l": 0, "r": 0, "t": 20, "b": 0},
)

energy_kinetic = 0.5 * np.sum(masses[None, :, None] * np.diff(positions, axis=0, prepend=positions[[0]]) ** 2, axis=(1, 2))

st.plotly_chart(fig3d, use_container_width=True)
st.line_chart({"Energie cinétique approximative": energy_kinetic}, x=time_points)

st.info(
    "Astuce: augmente légèrement une vitesse initiale (ex: +0.01) et compare les trajectoires, "
    "tu verras la sensibilité chaotique aux conditions initiales."
)
