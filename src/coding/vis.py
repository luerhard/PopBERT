from IPython.display import display
from IPython.display import HTML


def visualize_dataframe(
    df,
    id_col="example_id",
    text_col="text",
    color_col="label_color",
    hide_attrs=True,
):

    if hide_attrs:
        cols = [col for col in df.columns if col not in (color_col,)]
    else:
        cols = list(df.columns)

    spacing_row = (
        f"<tr style='background-color: transparent;' height='16px'>{'<td></td>'*len(cols)}</tr>\n"
    )

    table = "".join(
        ("<table>\n<thead>\n", *[f"<th>{col}</th>" for col in cols], "</thead>\n<tbody>\n"),
    )
    prev_id = None

    for col in df.itertuples():

        cur_id = getattr(col, id_col)
        if prev_id is None:
            prev_id = cur_id

        if cur_id != prev_id:
            prev_id = cur_id
            table += spacing_row

        row = "<tr>"

        color = getattr(col, color_col)
        for colname in cols:
            if colname != text_col:
                if not hide_attrs or colname not in (color_col,):
                    row += (
                        f"<td style='background-color: {color}40'>{str(getattr(col, colname))}</td>"
                    )
                continue

            text = getattr(col, text_col)

            row += "".join(
                (
                    "<td style='text-align: left;'>",
                    text,
                    "</td>",
                ),
            )

        row += "</tr>\n"
        table += row

    table += "</tbody>\n</table>"
    display(HTML(table))
