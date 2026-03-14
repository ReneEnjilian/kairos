FROM python:3.12

WORKDIR /app

COPY pyproject.toml .
COPY kairos ./kairos
COPY configs ./configs

RUN python -m pip install .

CMD ["python", "-m", "kairos.main"]