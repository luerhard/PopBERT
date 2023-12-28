import logging
import math
import re
import uuid
from collections import deque
from pathlib import Path

import sqlalchemy as sa
import torch
from fastapi import FastAPI
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

import src
import src.bert.io
from src.web.models import Base  # type: ignore
from src.web.models import Model  # type: ignore
from src.web.models import Prediction  # type: ignore
from src.web.models import Sample  # type: ignore
from src.web.sessionmanager import SessionManager

logfmt = "[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s"
logging.basicConfig(format=logfmt)
logger = logging.getLogger(__name__)


TOKENIZER = "deepset/gbert-large"
MODEL = "tmp/model_v8.model"

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)

templates = Jinja2Templates(directory="templates")
session_manager = SessionManager()


def assert_user(user):
    if user:
        return user
    return uuid.uuid4().hex


def get_chain(user) -> deque:
    global session_manager
    return session_manager.get(user, deque(maxlen=100))


def load_model():
    model = torch.load(src.PATH / MODEL)
    model.to("cpu")
    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return lambda x: tokenizer(x, padding=True, return_tensors="pt")


def load_engine():
    conn_string = f"sqlite:///{src.PATH / 'tmp/predictions.sqlite'}"
    engine = create_engine(conn_string)
    Base.metadata.create_all(engine)
    return engine


def translate_result(prediction: Prediction) -> list[tuple[str, float]]:
    dimensions = [
        "pop_antielite",
        "pop_pplcentr",
        # "souv_eliteless",
        # "souv_pplmore",
        # "ideol_left",
        # "ideol_right",
    ]

    results = []
    for dim in dimensions:
        val = getattr(prediction, dim)
        key = math.floor(val * 10)
        key = 9 if key == 10 else key
        results.append((f"perc{key}", round(val, 2)))

    return results


def predict(session, tokenizer, model, sample: Sample, model_id: int) -> Prediction:
    prediction = (
        session.query(Prediction)
        .filter(
            Prediction.sample_id == sample.id,
            Prediction.model_id == model_id,
        )
        .one_or_none()
    )

    if not prediction:
        encoding = tokenizer(sample.text)
        _, probas = model(**encoding)
        result = probas.mean(axis=1).detach().numpy()[0]

        logger.warn("New Prediction for Sample %s -- %s", sample.id, result)

        prediction = Prediction(
            sample_id=sample.id,
            model_id=model_id,
            pop_antielite=result[1],
            pop_pplcentr=result[2],
        )
        sample.predictions.append(prediction)
        session.add(sample)
        session.commit()
    else:
        logger.warn("Cached Prediction: %s for sample: %s", prediction.id, sample.id)

    return prediction


def get_sample(session, text: str) -> Sample:
    text = text.strip()

    sample = session.query(Sample).filter(Sample.text == text).one_or_none()
    if not sample:
        sample = Sample(text=text)
        session.add(sample)

    return sample


def get_model_info(session):
    version = re.search(r"(?<=_)(v.*?)(?=\.model$)", MODEL)
    if not version:
        raise Exception(f"No valid version string found in: {MODEL}")
    version = version.group(0)
    logger.info("Version extracted: %s", version)
    model = session.query(Model).filter(Model.version == version).one_or_none()
    if not model:
        model = Model(version=version, path=MODEL)
        session.add(model)
        session.commit()

    return version, model.id


engine = load_engine()
model = load_model()
tokenizer = load_tokenizer()

with Session(engine) as s:
    model_version, model_id = get_model_info(s)
    logger.info("Model ID found: %d", model_id)


@app.get("/high")
async def high(request: Request, sortby: str, n: int = 20, user: str | None = None):
    global engine
    global model
    global tokenizer
    global model_id
    global model_version

    logger.warn("Sorting by: %s", sortby)
    logger.warning("User: %s", user)
    user = assert_user(user)
    chain = get_chain(user)

    sorter = getattr(Prediction, sortby)

    with Session(engine) as session:
        samples = (
            session.query(Sample, Prediction)
            .join(
                Prediction,
                sa.and_(
                    Sample.id == Prediction.sample_id,
                    Prediction.model_id == model_id,
                ),
            )
            .order_by(sorter.desc())
            .limit(n)
        )
        chain.clear()
        for sample, pred in samples:
            result = translate_result(pred)
            chain.append({"message": sample.text, "result": result})

    session_manager[user] = chain
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "data": chain, "user": user, "version": model_version},
    )


@app.get("/random")
async def random(request: Request, n: int = 20, user: str | None = None):
    global engine
    global model
    global tokenizer
    global model_id
    global model_version

    logger.warning("User: %s", user)
    user = assert_user(user)
    chain = get_chain(user)

    chain.clear()
    with Session(engine) as session:
        samples = session.query(Sample).order_by(func.random()).limit(n)
        for sample in samples:
            pred = predict(session, tokenizer, model, sample, model_id=model_id)
            result = translate_result(pred)
            chain.appendleft({"message": sample.text, "result": result})

    session_manager[user] = chain
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "data": chain, "user": user, "version": model_version},
    )


@app.get("/")
async def root(request: Request, text: str = "", user: str | None = None):
    global engine
    global model
    global tokenizer
    global model_id
    global model_version

    logger.warning("User: %s", user)
    logger.warning("received text input: %s", text)
    user = assert_user(user)
    chain = get_chain(user)

    with Session(engine) as session:
        try:
            if len(chain) == 0:
                samples = session.query(Sample).order_by(func.random()).limit(10)
                for sample in samples:
                    pred = predict(session, tokenizer, model, sample, model_id=model_id)
                    result = translate_result(pred)
                    chain.appendleft({"message": sample.text, "result": result})

            try:
                prev_msg = chain[0]["message"]
            except IndexError:
                prev_msg = ""

            if text and prev_msg != text:
                sample = get_sample(session, text)
                pred = predict(session, tokenizer, model, sample, model_id=model_id)
                result = translate_result(pred)
                chain.appendleft({"message": sample.text, "result": result})

            session.commit()
        except Exception:
            session.rollback()
            raise

    session_manager[user] = chain
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "data": chain, "user": user, "version": model_version},
    )
