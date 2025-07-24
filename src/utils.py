def style_plotly_figure(fig, title: str) -> None:
    """Apply consistent styling to Plotly figures."""
    fig.update_layout(
        title_font_family="Poppins",
        font_family="Poppins",
        title=title,
        plot_bgcolor="#F9F6F0",
        paper_bgcolor="#F9F6F0",
        font=dict(color="#3E3E3E"),
        title_font_color="#3E3E3E",
        margin=dict(t=40, l=40, r=40, b=40),
        showlegend=True,
        legend=dict(
            bgcolor="#F9F6F0",
            bordercolor="#E8E6E1",
            borderwidth=1,
            font=dict(family="Poppins", color="#3E3E3E")
        )
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E8E6E1",
        linecolor="#3E3E3E",
        tickfont=dict(family="Poppins", color="#3E3E3E")
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E8E6E1",
        linecolor="#3E3E3E",
        tickfont=dict(family="Poppins", color="#3E3E3E")
    )
