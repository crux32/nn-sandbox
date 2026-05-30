from typing import Optional

import numpy as np
import plotly.graph_objects as go


def plot_true_vs_predicted_PFR(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_values: Optional[np.ndarray],
    true_style: Optional[dict] = None,
    pred_style: Optional[dict] = None,
    title: str = "True vs Predicted",
    xlabel: str = "V, [l]",
    ylabel: str = "X",
) -> go.Figure:
    """
    Plot true and predicted values with customizable styles.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        x_values: Optional x-axis values (default: index 0, 1, 2,...).
        true_style: Optional dict of Plotly style overrides for the true line.
        pred_style: Optional dict of Plotly style overrides for the predicted line.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        A plotly.graph_objects.Figure that can be shown with .show() or further customized.
    """
    # Default styles
    default_true_style = dict(
        name="True",
        mode="lines+markers",
        line=dict(color="blue", width=2, dash="solid"),
        marker=dict(symbol="circle", size=6, color="blue"),
    )
    default_pred_style = dict(
        name="Predicted",
        mode="lines+markers",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(symbol="x", size=6, color="red"),
    )

    # Merge with user overrides
    true_cfg = {**default_true_style, **(true_style or {})}
    pred_cfg = {**default_pred_style, **(pred_style or {})}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_true, **true_cfg))
    fig.add_trace(go.Scatter(x=x_values, y=y_pred, **pred_cfg))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode="x unified",
    )
    return fig
