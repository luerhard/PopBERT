import pandas as pd
from sqlalchemy import literal
from sqlalchemy.orm import Query

import src.db.models.doccano as m
from src.db.connect import make_engine


def labels_with_none(project):

    labels = (
        Query(m.ExamplesExample)
        .join(m.ProjectsProject)
        .join(m.LabelsSpan)
        .join(m.LabelTypesSpantype)
        .join(m.AuthUser, m.LabelsSpan.user_id == m.AuthUser.id)
        .filter(m.ProjectsProject.name == project)
        .with_entities(
            m.ExamplesExample.id.label("example_id"),
            m.ExamplesExample.text,
            m.LabelTypesSpantype.text.label("label"),
            m.AuthUser.username.label("label_by"),
            m.LabelTypesSpantype.background_color.label("label_color"),
            m.LabelsSpan.start_offset,
            m.LabelsSpan.end_offset,
        )
        .subquery()
    )

    empty = (
        Query(m.ExamplesExamplestate)
        .join(m.ExamplesExample, m.ExamplesExamplestate.example_id == m.ExamplesExample.id)
        .join(m.ProjectsProject, m.ExamplesExample.project_id == m.ProjectsProject.id)
        .join(m.AuthUser, m.ExamplesExamplestate.confirmed_by_id == m.AuthUser.id)
        .filter(
            m.ExamplesExamplestate.example_id.notin_(
                Query(labels).with_entities(labels.c.example_id),
            ),
            m.ProjectsProject.name == project,
        )
        .with_entities(
            m.ExamplesExamplestate.example_id.label("example_id"),
            m.ExamplesExample.text,
            literal("None").label("label"),
            m.AuthUser.username.label("label_by"),
            literal("#d3d3d3").label("label_color"),
            literal(0).label("start_offset"),
            literal(0).label("end_offset"),
        )
    )

    all_labels = Query(labels).union(empty)

    engine = make_engine("DOCCANO")
    df = pd.read_sql(all_labels.statement, engine)
    engine.dispose()
    df.columns = [col.lstrip("anon_1_") for col in df.columns]

    return df
